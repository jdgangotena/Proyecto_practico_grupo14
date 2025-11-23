"""
Entrenamiento de Modelo - Amazon Reviews Helpfulness Prediction (CORREGIDO)
- Filtra HelpfulnessRate == 0 antes del entrenamiento (configurable)
- Usa TF-IDF (configurable) + features numéricas
- Maneja desbalance con scale_pos_weight calculado automáticamente
- Encuentra umbral óptimo en validación (max F1) y lo usa para evaluar test
- Guarda modelo + metadata + 'latest' (pickle + metadata json)
"""

import os
import sys
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve
)
import lightgbm as lgb
import joblib
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin

# Plotly (opcional, igual que antes)
import plotly.express as px
import plotly.graph_objects as go

# Añadir el directorio scripts al path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# Rutas del proyecto (ajustables)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------- Configuración general (ajustable) ----------
TFIDF_MAX_FEATURES = 20000   # reducir si memoria es problema
TFIDF_NGRAM_RANGE = (1, 2)
USE_TFIDF = True             # si False, usa solo features numéricas
FILTER_ZERO_HELPFULNESS = True  # Eliminar reseñas con HelpfulnessRate == 0 antes del entrenamiento
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1   # fracción del train para validación (si se desea)
NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50
N_TRIALS = 20  # Optuna trials
# ------------------------------------------------------

# LGBMWrapper removed


class ReviewHelpfulnessModel:
    def __init__(self, use_tfidf=True, tfidf_max_features=20000, tfidf_ngram_range=(1,2)):
        self.model = None
        self.feature_columns = []        # lista completa de nombres de features (incl. TF-IDF)
        self.numeric_feature_names = []  # features numéricas originales
        self.tfidf = None
        self.use_tfidf = use_tfidf
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.train_params = {}
        self.model_metrics = {}
        self.best_threshold = 0.5

    def _prepare_text_features(self, texts):
        """
        Ajusta TF-IDF (fit_transform si es necesario) y devuelve matriz scipy.sparse.
        """
        if not self.use_tfidf:
            return None

        if self.tfidf is None:
            self.tfidf = TfidfVectorizer(
                max_features=self.tfidf_max_features,
                ngram_range=self.tfidf_ngram_range,
                min_df=3
            )
            X_tfidf = self.tfidf.fit_transform(texts.fillna("").astype(str))
        else:
            X_tfidf = self.tfidf.transform(texts.fillna("").astype(str))

        return X_tfidf

    def preparar_datos(
        self, df,
        text_column='CleanText',
        numeric_features=None,
        target='IsHelpful',
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        filter_zero_helpfulness=FILTER_ZERO_HELPFULNESS
    ):
        """
        Prepara matrices (sparse) y splits train/val/test.
        - Filtra HelpfulnessRate == 0 si se pide.
        - Construye TF-IDF y concatena con features numéricas (si hay).
        - Devuelve X_train, X_val, X_test (scipy.sparse) y y_train, y_val, y_test (arrays)
        - Guarda nombres de features para metadata.
        """
        print("\n--- PREPARANDO DATOS ---")

        if target not in df.columns:
            raise ValueError(f"Columna objetivo '{target}' no encontrada en el DataFrame")

        # Filtrar HelpfulnessRate == 0 si corresponde
        if filter_zero_helpfulness:
            if 'HelpfulnessRate' not in df.columns:
                print("⚠️ HelpfulnessRate no está en df — no se aplicará el filtro")
            else:
                before = len(df)
                df = df[df['HelpfulnessRate'] > 0].copy()
                after = len(df)
                print(f"Filtrado HelpfulnessRate==0: {before - after} filas eliminadas ({(before-after)/before*100:.1f}%)")

        # Text column check
        if self.use_tfidf and text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' requerida para TF-IDF no encontrada.")

        # Default numeric features if no list provided (toma columnas que existan)
        if numeric_features is None:
            candidate_numeric = [
                'char_count', 'word_count', 'avg_word_length', 'sentence_count', 'words_per_sentence',
                'exclamation_count', 'question_count', 'uppercase_word_count', 'lexical_diversity',
                'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'textblob_polarity', 'textblob_subjectivity',
                'digit_ratio', 'specificity_score', 'has_comparison', 'personal_experience_score', 'price_mention',
                'flesch_reading_ease', 'gunning_fog', 'paragraph_count', 'bullet_point_count'
            ]
            numeric_features = [c for c in candidate_numeric if c in df.columns]

        self.numeric_feature_names = numeric_features.copy()
        print(f"Features numéricas usadas: {len(self.numeric_feature_names)}")

        # Split inicial (train+val) / test con estratificación
        X_tmp = df.copy()
        y = X_tmp[target].astype(int)

        # Si hay muy pocas positivas, advertir
        pos_count = int((y == 1).sum())
        neg_count = int((y == 0).sum())
        total = len(y)
        print(f"Distribución objetivo (total={total}): positivos={pos_count}, negativos={neg_count}")
        if pos_count < 30:
            print("⚠️ Muy pocas muestras positivas (<30). Los resultados serán inestables.")

        X_trainval_df, X_test_df, y_trainval, y_test = train_test_split(
            X_tmp, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Dentro de trainval, separar validation si val_size > 0
        if val_size and val_size > 0:
            val_frac_of_trainval = val_size / (1 - test_size)
            X_train_df, X_val_df, y_train, y_val = train_test_split(
                X_trainval_df, y_trainval, test_size=val_frac_of_trainval,
                random_state=random_state, stratify=y_trainval
            )
        else:
            X_train_df = X_trainval_df
            y_train = y_trainval
            X_val_df = None
            y_val = None

        print(f"Train: {len(X_train_df)}, Val: {len(X_val_df) if X_val_df is not None else 0}, Test: {len(X_test_df)}")

        # Preparar numeric matrices
        X_train_num = X_train_df[self.numeric_feature_names].fillna(0).values if self.numeric_feature_names else None
        X_val_num = X_val_df[self.numeric_feature_names].fillna(0).values if (X_val_df is not None and self.numeric_feature_names) else None
        X_test_num = X_test_df[self.numeric_feature_names].fillna(0).values if self.numeric_feature_names else None

        # Preparar TF-IDF
        if self.use_tfidf:
            X_train_text = self._prepare_text_features(X_train_df[text_column])
            X_val_text = self._prepare_text_features(X_val_df[text_column]) if X_val_df is not None else None
            X_test_text = self._prepare_text_features(X_test_df[text_column])

            # Concatenar numeric + tfidf (usar matrices sparse)
            parts_train = []
            parts_val = []
            parts_test = []

            if X_train_num is not None and X_train_num.size > 0:
                parts_train.append(sparse.csr_matrix(X_train_num))
            parts_train.append(X_train_text)

            if X_val_df is not None:
                if X_val_num is not None and X_val_num.size > 0:
                    parts_val.append(sparse.csr_matrix(X_val_num))
                parts_val.append(X_val_text)

            if X_test_num is not None and X_test_num.size > 0:
                parts_test.append(sparse.csr_matrix(X_test_num))
            parts_test.append(X_test_text)

            X_train = hstack(parts_train).tocsr()
            X_val = hstack(parts_val).tocsr() if X_val_df is not None else None
            X_test = hstack(parts_test).tocsr()
            
            # Construir nombres de features (numeric + tfidf names)
            tfidf_names = list(self.tfidf.get_feature_names_out()) if self.tfidf is not None else []
            self.feature_columns = self.numeric_feature_names + tfidf_names

        else:
            # Solo numeric (numpy arrays -> convertir a sparse para LGB compat)
            X_train = sparse.csr_matrix(X_train_num) if X_train_num is not None else None
            X_val = sparse.csr_matrix(X_val_num) if X_val_num is not None else None
            X_test = sparse.csr_matrix(X_test_num) if X_test_num is not None else None
            self.feature_columns = self.numeric_feature_names.copy()

        # Convertir y a arrays
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val) if y_val is not None else None
        y_test = np.asarray(y_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def entrenar(self, X_train, y_train, X_val=None, y_val=None, params=None,
                 num_boost_round=NUM_BOOST_ROUND, early_stopping_rounds=EARLY_STOPPING_ROUNDS):
        """
        Entrena LightGBM con manejo de clases desbalanceadas.
        Acepta X como scipy.sparse.
        """
        print("\n--- ENTRENANDO LightGBM ---")

        # Calcular scale_pos_weight automáticamente
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        if pos == 0:
            raise ValueError("No hay ejemplos positivos en y_train. Imposible entrenar.")
        scale_pos_weight = neg / pos

        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbosity': -1,
            'seed': RANDOM_STATE,
            # Manejo de desbalance:
            'scale_pos_weight': scale_pos_weight
        }

        if params:
            default_params.update(params)

        self.train_params = default_params.copy()
        print(f"Parámetros de entrenamiento (scale_pos_weight={scale_pos_weight:.2f}):")
        for k, v in default_params.items():
            print(f"  - {k}: {v}")

        # Crear dataset para LightGBM (soporta scipy.sparse)
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_columns)

        valid_sets = [train_data]
        valid_names = ['train']
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=self.feature_columns)
            valid_sets.append(val_data)
            valid_names.append('valid')

            self.model = lgb.train(
                default_params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=50
            )
        else:
            self.model = lgb.train(
                default_params,
                train_data,
                num_boost_round=200
            )

        print("✓ Entrenamiento finalizado")
        return self.model

    def optimizar_hiperparametros(self, X_train, y_train, X_val, y_val):
        """Usa Optuna para encontrar los mejores hiperparámetros."""
        print("\n--- OPTIMIZANDO HIPERPARÁMETROS (OPTUNA) ---")
        
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            }
            
            # Entrenar modelo rápido
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            callbacks = [
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(period=0)
            ]

            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=callbacks
            )
            
            # Maximizar AUC en validación
            preds = model.predict(X_val)
            auc = roc_auc_score(y_val, preds)
            return auc

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=N_TRIALS)
        
        print(f"Mejores parámetros: {study.best_params}")
        return study.best_params

    def entrenar_calibrado(self, X_train, y_train, X_val, y_val, params=None):
        """Entrena modelo y luego calibra probabilidades con Isotonic Regression."""
        print("\n--- ENTRENANDO MODELO CALIBRADO ---")
        
        # Usar LGBMClassifier estándar
        # Asegurar que params no tenga duplicados con kwargs de LGBMClassifier
        if params is None: params = {}
        
        clf = lgb.LGBMClassifier(**params, n_estimators=NUM_BOOST_ROUND, verbose=-1)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=0)
        ]
        
        print("1. Entrenando modelo base...")
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='binary_logloss',
            callbacks=callbacks
        )
        
        self.model = clf # Guardar referencia
        
        print("2. Calibrando probabilidades (Isotonic)...")
        self.calibrated_model = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
        self.calibrated_model.fit(X_val, y_val)
        
        print("✓ Modelo calibrado listo")
        return self.calibrated_model

    def _find_best_threshold(self, y_true, y_proba):
        """
        Encuentra threshold que maximiza F1 en (y_true, y_proba) usando precision_recall_curve.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        # thresholds length = len(precision)-1
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
        best_idx = np.nanargmax(f1_scores)
        # best_idx puede corresponder a precision[best_idx], thresholds[best_idx-1] -> cuidamos indices
        if best_idx == 0:
            best_threshold = 0.5
        else:
            best_threshold = thresholds[best_idx - 1]
        return float(best_threshold), float(np.nanmax(f1_scores))

    def evaluar(self, X_test, y_test, use_threshold=None, X_val=None, y_val=None):
        """
        Evalúa el modelo. Si se proporciona X_val,y_val, calcula threshold óptimo en validación.
        De lo contrario, usa threshold pasado o 0.5.
        Retorna métricas y guarda en self.model_metrics y self.best_threshold.
        """
        print("\n--- EVALUANDO MODELO ---")
        if self.model is None:
            raise ValueError("Modelo no entrenado")

        # Probabilidades
        if hasattr(self, 'calibrated_model') and self.calibrated_model is not None:
            y_pred_proba = self.calibrated_model.predict_proba(X_test)[:, 1]
        elif hasattr(self.model, "predict_proba"):
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = self.model.predict(X_test)
        # Determinar umbral óptimo
        if X_val is not None and y_val is not None:
            y_val_proba = self.model.predict(X_val)
            best_thr, best_f1 = self._find_best_threshold(y_val, y_val_proba)
            print(f"Umbral óptimo calculado en validación (max F1={best_f1:.4f}): {best_thr:.4f}")
            self.best_threshold = best_thr
        elif use_threshold is not None:
            self.best_threshold = use_threshold
            print(f"Usando umbral provisto: {self.best_threshold:.4f}")
        else:
            self.best_threshold = 0.5
            print("Usando umbral por defecto: 0.5")

        y_pred = (y_pred_proba >= self.best_threshold).astype(int)

        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
        }

        self.model_metrics = metrics

        print("\nMétricas de evaluación (usando umbral = {:.4f}):".format(self.best_threshold))
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred, target_names=['No Útil', 'Útil'], zero_division=0))

        print("\nMatriz de confusión:")
        print(confusion_matrix(y_test, y_pred))

        return metrics, y_pred_proba, y_pred
    
    def predecir(self, X):
        """
        Devuelve probabilidades de que la reseña sea útil (clase 1).
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado aún.")

        # Si el modelo tiene predict_proba (LightGBM, XGB, RandomForest, etc.)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]

        # Si tenemos modelo calibrado, usarlo
        if hasattr(self, 'calibrated_model') and self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)[:, 1]

        # Si el modelo tiene predict_proba (LightGBM, XGB, RandomForest, etc.)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]

        # Si solo tiene predict (algunos modelos de regresión)
        preds = self.model.predict(X)
        return preds


    def obtener_importancia_features(self, top_n=50):
        """
        Obtiene importancia de features desde el modelo LightGBM.
        Devuelve DataFrame con columnas 'feature' e 'importance' ordenado desc.
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado")

        importances = self.model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        return feature_importance.head(top_n)

    def guardar_modelo(self, nombre='review_helpfulness_model'):
        """
        Guarda modelo (pickle) y metadata (json). También guarda 'latest'.
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODEL_DIR, f"{nombre}_{timestamp}.pkl")
        metadata_path = os.path.join(MODEL_DIR, f"{nombre}_{timestamp}_metadata.json")

        # Guardar modelo con joblib/pickle
        # Guardar el calibrado si existe, sino el base
        model_to_save = self.calibrated_model if hasattr(self, 'calibrated_model') else self.model
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_to_save, f)

        metadata = {
            'timestamp': timestamp,
            'model_type': 'LightGBM',
            'feature_columns': self.feature_columns,
            'numeric_features': self.numeric_feature_names,
            'use_tfidf': self.use_tfidf,
            'tfidf_max_features': self.tfidf_max_features,
            'tfidf_ngram_range': self.tfidf_ngram_range,
            'best_threshold': self.best_threshold,
            'metrics': self.model_metrics,
            'params': self.train_params,
            'num_features': len(self.feature_columns)
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # guardar latest
        latest_model_path = os.path.join(MODEL_DIR, f"{nombre}_latest.pkl")
        latest_metadata_path = os.path.join(MODEL_DIR, f"{nombre}_latest_metadata.json")
        with open(latest_model_path, 'wb') as f:
            pickle.dump(model_to_save, f)
        with open(latest_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Modelo guardado: {model_path}")
        print(f"✓ Metadata guardada: {metadata_path}")
        return model_path, metadata_path

    @classmethod
    def cargar_modelo(cls, model_path, load_tfidf=True):
        """
        Carga modelo y metadata. NOTA: if TF-IDF was used, deberías reconstruir/restore the vectorizer separately.
        """
        inst = cls()
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
            
        # Detectar si es calibrado o lgb nativo
        if isinstance(loaded_model, CalibratedClassifierCV):
            inst.calibrated_model = loaded_model
            inst.model = loaded_model.base_estimator.model # Acceder al lgb interno
        else:
            inst.model = loaded_model

        # cargar metadata
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
            inst.feature_columns = meta.get('feature_columns', [])
            inst.numeric_feature_names = meta.get('numeric_features', [])
            inst.use_tfidf = meta.get('use_tfidf', False)
            inst.tfidf_max_features = meta.get('tfidf_max_features', None)
            inst.tfidf_ngram_range = tuple(meta.get('tfidf_ngram_range', (1,2)))
            inst.best_threshold = meta.get('best_threshold', 0.5)
            inst.model_metrics = meta.get('metrics', {})
            inst.train_params = meta.get('params', {})

        print(f"✓ Modelo cargado desde: {model_path}")
        return inst


# ----------------- Utilidades para gráficos (similar a tu versión) -----------------
def crear_graficos_evaluacion(y_test, y_pred_proba, feature_importance, save=True):
    print("\n--- GENERANDO GRÁFICOS ---")
    try:
        fpr, tpr, _ = roc_curve = None
        # ROC
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                     name=f'ROC curve (AUC = {roc_auc:.3f})'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                     line=dict(dash='dash'), name='Random'))
        fig_roc.update_layout(title='ROC Curve', xaxis_title='FPR', yaxis_title='TPR', template='plotly_white')
        if save:
            fig_roc.write_html(os.path.join(PLOTS_DIR, 'roc_curve.html'))
            print("✓ Curva ROC guardada")

        # Feature importance
        fig_imp = px.bar(feature_importance.head(30), x='importance', y='feature', orientation='h',
                         title='Top Features Importance')
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_white')
        if save:
            fig_imp.write_html(os.path.join(PLOTS_DIR, 'feature_importance.html'))
            print("✓ Importancia de features guardada")

        # Distribución probabilidades (No Útil vs Útil)
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=y_pred_proba[np.array(y_test)==0], name='No Útil', opacity=0.7, nbinsx=50))
        fig_dist.add_trace(go.Histogram(x=y_pred_proba[np.array(y_test)==1], name='Útil', opacity=0.7, nbinsx=50))
        fig_dist.update_layout(barmode='overlay', title='Distribución probabilidad predicha', template='plotly_white')
        if save:
            fig_dist.write_html(os.path.join(PLOTS_DIR, 'probability_distribution.html'))
            print("✓ Distribución de probabilidades guardada")

        return fig_roc, fig_imp, fig_dist
    except Exception as e:
        print("Error generando gráficos:", e)
        return None, None, None

# ----------------- Script principal -----------------
if __name__ == "__main__":
    print("="*60)
    print("ENTRENAMIENTO CORREGIDO - PREDICCIÓN DE UTILIDAD DE RESEÑAS")
    print("="*60)

    data_path = os.path.join(DATA_DIR, "amazon_reviews_with_features.csv")
    if not os.path.exists(data_path):
        print(f"Error: no se encuentra {data_path}. Genera el dataset con nlp_features.py primero.")
        sys.exit(1)

    print(f"Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path)
    print(f"{len(df)} filas cargadas")

    # Instanciar modelo
    model = ReviewHelpfulnessModel(use_tfidf=USE_TFIDF, tfidf_max_features=TFIDF_MAX_FEATURES,
                                   tfidf_ngram_range=TFIDF_NGRAM_RANGE)

    # Preparar datos (filtra HelpfulnessRate==0 si FILTER_ZERO_HELPFULNESS=True)
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = model.preparar_datos(
            df,
            text_column='CleanText',
            numeric_features=None,  # usa lista por defecto
            target='IsHelpful',
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            random_state=RANDOM_STATE,
            filter_zero_helpfulness=FILTER_ZERO_HELPFULNESS
        )
    except Exception as e:
        print("Error preparando datos:", e)
        raise

    # Optimizar
    best_params = model.optimizar_hiperparametros(X_train, y_train, X_val, y_val)
    
    # Entrenar y Calibrar
    model.entrenar_calibrado(X_train, y_train, X_val, y_val, params=best_params)

    # Evaluar (usa validación para calcular umbral óptimo)
    metrics, y_pred_proba, y_pred = model.evaluar(X_test, y_test, X_val=X_val, y_val=y_val)

    # Importancia de features
    try:
        fi = model.obtener_importancia_features(top_n=50)
        print("\n--- TOP 10 FEATURES ---")
        print(fi.head(10).to_string(index=False))
    except Exception as e:
        print("No se pudo obtener importancia de features:", e)
        fi = pd.DataFrame({'feature':[], 'importance':[]})

    # Crear gráficos
    crear_graficos_evaluacion(y_test, y_pred_proba, fi, save=True)

    # Guardar modelo y metadata
    model.guardar_modelo('review_helpfulness_model')

    print("\n✓ Entrenamiento completado correctamente")
    print("Métricas finales:")
    for k,v in metrics.items():
        print(f"  {k}: {v:.4f}")
