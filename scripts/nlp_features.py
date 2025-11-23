"""
Extracción de Características NLP - Amazon Reviews
Extrae características de texto para predecir la utilidad de reseñas.
"""

import pandas as pd
import numpy as np
import re
import nltk
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import sys

# Añadir el directorio scripts al path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# ================================
# DESCARGA DE RECURSOS NLTK
# ================================

# VADER lexicon
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Punkt tokenizer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# 🔥 FIX CRÍTICO: Descargar punkt_tab (nuevo requerimiento de nltk)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class NLPFeatureExtractor:
    """Extrae características NLP de reseñas de texto."""

    def __init__(self):
        """Inicializa el extractor con los modelos necesarios."""
        self.vader = SentimentIntensityAnalyzer()

    def extraer_longitud_texto(self, text):
        """Calcula métricas de longitud del texto."""
        features = {}

        # Longitud en caracteres
        features['char_count'] = len(text)

        # Longitud en palabras
        words = text.split()
        features['word_count'] = len(words)

        # Longitud promedio de palabras
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0

        # Número de oraciones
        sentences = nltk.sent_tokenize(text)
        features['sentence_count'] = len(sentences)

        # Palabras por oración
        features['words_per_sentence'] = (
            features['word_count'] / features['sentence_count']
            if features['sentence_count'] > 0 else 0
        )

        return features

    def extraer_caracteristicas_lexicas(self, text):
        """Extrae características léxicas del texto."""
        features = {}

        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')

        words = text.split()
        features['uppercase_word_count'] = sum(
            1 for w in words if w.isupper() and len(w) > 1
        )

        unique_words = set(words)
        features['lexical_diversity'] = len(unique_words) / len(words) if words else 0

        return features

    def extraer_sentimiento_vader(self, text):
        """Analiza sentimiento usando VADER."""
        scores = self.vader.polarity_scores(text)
        return {
            'vader_neg': scores['neg'],
            'vader_neu': scores['neu'],
            'vader_pos': scores['pos'],
            'vader_compound': scores['compound']
        }

    def extraer_sentimiento_textblob(self, text):
        """Analiza sentimiento usando TextBlob."""
        try:
            blob = TextBlob(text)
            return {
                'textblob_polarity': blob.sentiment.polarity,
                'textblob_subjectivity': blob.sentiment.subjectivity
            }
        except:
            return {
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0
            }

    def extraer_especificidad_alimentos(self, text):
        """Detecta vocabulario específico de alimentos."""
        text_lower = text.lower()

        taste_words = ['sweet', 'salty', 'bitter', 'sour', 'umami', 'flavor', 'taste', 'spicy', 'bland']
        texture_words = ['crunchy', 'soft', 'chewy', 'tender', 'crispy', 'smooth', 'creamy', 'hard']
        quality_words = ['fresh', 'stale', 'rancid', 'expired', 'organic', 'natural', 'premium']

        taste_count = sum(text_lower.count(word) for word in taste_words)
        texture_count = sum(text_lower.count(word) for word in texture_words)
        quality_count = sum(text_lower.count(word) for word in quality_words)

        return {
            'specificity_score': taste_count + texture_count + quality_count
        }

    def extraer_comparaciones(self, text):
        """Detecta si la reseña hace comparaciones."""
        text_lower = text.lower()
        comparison_words = ['than', 'better', 'worse', 'compared', 'versus', 'vs', 'instead', 'alternative', 'similar']
        comparison_count = sum(text_lower.count(word) for word in comparison_words)
        return {
            'has_comparison': 1 if comparison_count > 0 else 0
        }

    def extraer_experiencia_personal(self, text):
        """Detecta indicadores de experiencia personal."""
        text_lower = text.lower()

        personal_pronouns = ['i ', 'my ', 'me ', 'we ', 'our ', "i've", "i'll"]
        time_indicators = ['days', 'weeks', 'months', 'years', 'always', 'daily', 'every', 'usually']

        personal_count = sum(text_lower.count(word) for word in personal_pronouns)
        time_count = sum(text_lower.count(word) for word in time_indicators)

        return {
            'personal_experience_score': personal_count + time_count
        }

    def extraer_menciones_precio(self, text):
        """Detecta menciones de precio."""
        text_lower = text.lower()

        price_words = [
            'price', 'cost', 'expensive', 'cheap',
            'worth', 'value', 'money', 'overpriced', 'affordable'
        ]
        price_count = sum(text_lower.count(word) for word in price_words)

        return {
            'price_mention': 1 if price_count > 0 else 0
        }

    def extraer_caracteristicas_adicionales(self, text, score=None):
        """Extrae características adicionales útiles."""
        return {
            'digit_ratio': sum(c.isdigit() for c in text) / len(text) if text else 0
        }

    def extraer_todas_caracteristicas(self, text, score=None):
        """Extrae todas las características NLP del texto."""
        features = {}

        # Longitud y estructura
        features.update(self.extraer_longitud_texto(text))
        features.update(self.extraer_caracteristicas_lexicas(text))

        # Análisis de sentimiento
        features.update(self.extraer_sentimiento_vader(text))
        features.update(self.extraer_sentimiento_textblob(text))

        # Características adicionales
        features.update(self.extraer_caracteristicas_adicionales(text, score))
        features.update(self.extraer_especificidad_alimentos(text))
        features.update(self.extraer_comparaciones(text))
        features.update(self.extraer_experiencia_personal(text))
        features.update(self.extraer_menciones_precio(text))

        return features


def procesar_dataset(df, text_column='CleanText', score_column='Score'):
    """Procesa todo el dataset y extrae características NLP."""
    print("\n--- EXTRAYENDO CARACTERÍSTICAS NLP ---")
    print(f"Procesando {len(df)} reseñas...")

    extractor = NLPFeatureExtractor()
    features_list = []

    for idx, row in df.iterrows():
        text = str(row[text_column])
        score = row[score_column] if score_column in df.columns else None
        features = extractor.extraer_todas_caracteristicas(text, score)
        features_list.append(features)

        if (idx + 1) % 1000 == 0:
            print(f"  Procesadas {idx + 1} reseñas...")

    features_df = pd.DataFrame(features_list)
    df_con_features = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    print(f"✓ Extracción completada. {len(features_df.columns)} características añadidas")
    return df_con_features


def obtener_estadisticas_features(df):
    """Muestra estadísticas de las características extraídas."""
    print("\n--- ESTADÍSTICAS DE CARACTERÍSTICAS ---")

    feature_columns = [
        'char_count', 'word_count', 'sentence_count',
        'vader_compound', 'textblob_polarity',
        'lexical_diversity'
    ]

    for col in feature_columns:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  Media: {df[col].mean():.3f}")
            print(f"  Mediana: {df[col].median():.3f}")
            print(f"  Std: {df[col].std():.3f}")


def analizar_correlaciones(df, target='IsHelpful'):
    """Analiza correlaciones entre características y la utilidad."""
    print(f"\n--- CORRELACIONES CON {target} ---")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if target in df.columns:
        correlations = df[numeric_cols].corrwith(df[target]).sort_values(ascending=False)

        print("\nTop 10 características más correlacionadas:")
        print(correlations.head(10))

        print("\nTop 10 características menos correlacionadas:")
        print(correlations.tail(10))

        return correlations

    return None


if __name__ == "__main__":
    print("="*60)
    print("EXTRACCIÓN DE CARACTERÍSTICAS NLP")
    print("="*60)

    data_path = os.path.join(SCRIPT_DIR, "..", "data", "amazon_reviews_prepared.csv")

    if not os.path.exists(data_path):
        print(f"Error: No se encuentra {data_path}")
        print("Ejecuta primero limpieza.py para generar el dataset preprocesado")
        sys.exit(1)

    print(f"\nCargando datos desde: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ {len(df)} reseñas cargadas")

    df_con_features = procesar_dataset(df, text_column='CleanText', score_column='Score')

    obtener_estadisticas_features(df_con_features)

    if 'IsHelpful' in df_con_features.columns:
        correlaciones = analizar_correlaciones(df_con_features, target='IsHelpful')

    output_path = os.path.join(SCRIPT_DIR, "..", "data", "amazon_reviews_with_features.csv")
    df_con_features.to_csv(output_path, index=False)
    print(f"\n✓ Dataset con características guardado en: {output_path}")

    print("\n--- MUESTRA DE CARACTERÍSTICAS ---")
    feature_cols = ['word_count', 'sentence_count', 'vader_compound', 'textblob_polarity', 'IsHelpful']
    existing_cols = [col for col in feature_cols if col in df_con_features.columns]
    print(df_con_features[existing_cols].head(10))

    print("\n✓ Extracción de características completada exitosamente")
