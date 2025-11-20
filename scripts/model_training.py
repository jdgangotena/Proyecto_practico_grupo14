"""
Entrenamiento de Modelo - Amazon Reviews Helpfulness Prediction
Entrena un modelo LightGBM para predecir la utilidad de reseñas.
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve
)
import lightgbm as lgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Añadir el directorio scripts al path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# Directorios
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "plots")

# Crear directorios si no existen
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


class ReviewHelpfulnessModel:
    """Modelo para predecir la utilidad de reseñas."""

    def __init__(self):
        """Inicializa el modelo."""
        self.model = None
        self.feature_columns = None
        self.model_metrics = {}

    def preparar_datos(self, df, target='IsHelpful', test_size=0.2, random_state=42):
        """
        Prepara los datos para entrenamiento.

        Args:
            df: DataFrame con características
            target: Columna objetivo
            test_size: Proporción de datos para test
            random_state: Semilla aleatoria

        Returns:
            X_train, X_test, y_train, y_test
        """
        print("\n--- PREPARANDO DATOS ---")

        # Seleccionar SOLO características objetivas (sin sentimiento ni score)
        # Esto permite que el modelo valore reseñas detalladas independientemente
        # de si son positivas o negativas
        feature_cols = [
            # Características de longitud (5)
            'char_count', 'word_count', 'avg_word_length', 'sentence_count',
            'words_per_sentence',
            # Características léxicas (4)
            'exclamation_count', 'question_count',
            'uppercase_word_count', 'lexical_diversity',
            # Características básicas (1)
            'digit_ratio',
            # Características específicas de dominio - Amazon Food Reviews (4)
            'specificity_score', 'has_comparison',
            'personal_experience_score', 'price_mention'
        ]

        # Verificar que las columnas existen
        self.feature_columns = [col for col in feature_cols if col in df.columns]
        print(f"Características seleccionadas: {len(self.feature_columns)}")

        # Verificar que el target existe
        if target not in df.columns:
            raise ValueError(f"La columna objetivo '{target}' no existe en el DataFrame")

        # Preparar X e y
        X = df[self.feature_columns].copy()
        y = df[target].copy()

        # Manejar valores faltantes
        X = X.fillna(0)

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Train set: {len(X_train)} muestras")
        print(f"Test set: {len(X_test)} muestras")
        print(f"Distribución de clases en train: {y_train.value_counts().to_dict()}")
        print(f"Distribución de clases en test: {y_test.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test

    def entrenar(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """
        Entrena el modelo LightGBM.

        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de entrenamiento
            X_val: Características de validación (opcional)
            y_val: Etiquetas de validación (opcional)
            params: Hiperparámetros personalizados

        Returns:
            Modelo entrenado
        """
        print("\n--- ENTRENANDO MODELO LightGBM ---")

        # Parámetros por defecto
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }

        # Actualizar con parámetros personalizados
        if params:
            default_params.update(params)

        # Crear datasets de LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)

        # Entrenar modelo
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            self.model = lgb.train(
                default_params,
                train_data,
                num_boost_round=200,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=20)]
            )
        else:
            self.model = lgb.train(
                default_params,
                train_data,
                num_boost_round=100
            )

        print("✓ Entrenamiento completado")

        return self.model

    def evaluar(self, X_test, y_test):
        """
        Evalúa el modelo en el conjunto de test.

        Args:
            X_test: Características de test
            y_test: Etiquetas de test

        Returns:
            dict: Métricas de evaluación
        """
        print("\n--- EVALUANDO MODELO ---")

        # Predicciones
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Calcular métricas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        self.model_metrics = metrics

        print("\nMétricas de evaluación:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred, target_names=['No Útil', 'Útil']))

        print("\nMatriz de confusión:")
        print(confusion_matrix(y_test, y_pred))

        return metrics

    def obtener_importancia_features(self):
        """
        Obtiene la importancia de las características.

        Returns:
            DataFrame con importancia de características
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado")

        importances = self.model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return feature_importance

    def predecir(self, X):
        """
        Realiza predicciones con el modelo.

        Args:
            X: Características

        Returns:
            Probabilidades de predicción
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado")

        return self.model.predict(X)

    def guardar_modelo(self, nombre='review_helpfulness_model'):
        """
        Guarda el modelo entrenado.

        Args:
            nombre: Nombre del archivo del modelo
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODEL_DIR, f"{nombre}_{timestamp}.pkl")
        metadata_path = os.path.join(MODEL_DIR, f"{nombre}_{timestamp}_metadata.json")

        # Guardar modelo
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        # Guardar metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'metrics': self.model_metrics,
            'timestamp': timestamp
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Modelo guardado en: {model_path}")
        print(f"✓ Metadata guardada en: {metadata_path}")

        # También guardar una versión "latest"
        latest_model_path = os.path.join(MODEL_DIR, f"{nombre}_latest.pkl")
        latest_metadata_path = os.path.join(MODEL_DIR, f"{nombre}_latest_metadata.json")

        with open(latest_model_path, 'wb') as f:
            pickle.dump(self.model, f)

        with open(latest_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Modelo 'latest' guardado en: {latest_model_path}")

        return model_path

    @classmethod
    def cargar_modelo(cls, model_path):
        """
        Carga un modelo guardado.

        Args:
            model_path: Ruta al archivo del modelo

        Returns:
            Instancia de ReviewHelpfulnessModel
        """
        instance = cls()

        with open(model_path, 'rb') as f:
            instance.model = pickle.load(f)

        # Cargar metadata
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                instance.feature_columns = metadata.get('feature_columns', [])
                instance.model_metrics = metadata.get('metrics', {})

        print(f"✓ Modelo cargado desde: {model_path}")

        return instance


def crear_graficos_evaluacion(y_test, y_pred_proba, feature_importance, save=True):
    """
    Crea gráficos de evaluación del modelo usando Plotly.

    Args:
        y_test: Etiquetas verdaderas
        y_pred_proba: Probabilidades predichas
        feature_importance: DataFrame con importancia de características
        save: Si True, guarda los gráficos
    """
    print("\n--- GENERANDO GRÁFICOS ---")

    # 1. Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.3f})',
        line=dict(color='darkorange', width=2)
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='navy', width=2, dash='dash')
    ))
    fig_roc.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_white'
    )

    if save:
        fig_roc.write_html(os.path.join(PLOTS_DIR, 'roc_curve.html'))
        print("✓ Curva ROC guardada")

    # 2. Importancia de características
    top_features = feature_importance.head(15)

    fig_importance = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 15 Características Más Importantes',
        labels={'importance': 'Importancia', 'feature': 'Característica'},
        color='importance',
        color_continuous_scale='Viridis'
    )
    fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'}, template='plotly_white')

    if save:
        fig_importance.write_html(os.path.join(PLOTS_DIR, 'feature_importance.html'))
        print("✓ Importancia de características guardada")

    # 3. Distribución de probabilidades predichas
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=y_pred_proba[y_test == 0],
        name='No Útil',
        opacity=0.7,
        nbinsx=50
    ))
    fig_dist.add_trace(go.Histogram(
        x=y_pred_proba[y_test == 1],
        name='Útil',
        opacity=0.7,
        nbinsx=50
    ))
    fig_dist.update_layout(
        title='Distribución de Probabilidades Predichas',
        xaxis_title='Probabilidad de ser Útil',
        yaxis_title='Frecuencia',
        barmode='overlay',
        template='plotly_white'
    )

    if save:
        fig_dist.write_html(os.path.join(PLOTS_DIR, 'probability_distribution.html'))
        print("✓ Distribución de probabilidades guardada")

    return fig_roc, fig_importance, fig_dist


if __name__ == "__main__":
    print("="*60)
    print("ENTRENAMIENTO DE MODELO - PREDICCIÓN DE UTILIDAD DE RESEÑAS")
    print("="*60)

    # Cargar datos con características
    data_path = os.path.join(SCRIPT_DIR, "..", "data", "amazon_reviews_with_features.csv")

    if not os.path.exists(data_path):
        print(f"Error: No se encuentra {data_path}")
        print("Ejecuta primero nlp_features.py para generar el dataset con características")
        sys.exit(1)

    print(f"\nCargando datos desde: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ {len(df)} reseñas cargadas")

    # Crear instancia del modelo
    model = ReviewHelpfulnessModel()

    # Preparar datos
    X_train, X_test, y_train, y_test = model.preparar_datos(df, target='IsHelpful')

    # Entrenar modelo
    model.entrenar(X_train, y_train)

    # Evaluar modelo
    metrics = model.evaluar(X_test, y_test)

    # Obtener importancia de características
    feature_importance = model.obtener_importancia_features()
    print("\n--- TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES ---")
    print(feature_importance.head(10))

    # Crear gráficos
    y_pred_proba = model.predecir(X_test)
    crear_graficos_evaluacion(y_test, y_pred_proba, feature_importance, save=True)

    # Guardar modelo
    model.guardar_modelo('review_helpfulness_model')

    print("\n✓ Entrenamiento completado exitosamente")
