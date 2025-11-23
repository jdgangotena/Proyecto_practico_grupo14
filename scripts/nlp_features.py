"""
Extracción de Características NLP - Amazon Reviews
Extrae características de texto para predecir la utilidad de reseñas.
"""

import pandas as pd
import numpy as np
import nltk
from textblob import TextBlob
import textstat
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import sys

# Añadir el directorio scripts al path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# ================================
# DESCARGA DE RECURSOS NLTK
# ================================

def descargar_recurso(resource, path):
    """Evita errores silenciosos al descargar recursos de NLTK."""
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)

descargar_recurso('vader_lexicon', 'sentiment/vader_lexicon.zip')
descargar_recurso('punkt', 'tokenizers/punkt')
descargar_recurso('punkt_tab', 'tokenizers/punkt_tab')


# ================================
#  CLASE DE EXTRACCIÓN DE FEATURES
# ================================

class NLPFeatureExtractor:
    """Extrae características NLP de reseñas de texto."""

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    # -------- LONGITUD DEL TEXTO -------- #
    def extraer_longitud_texto(self, text):
        words = text.split()
        sentences = nltk.sent_tokenize(text)

        return {
            'char_count': len(text),
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'sentence_count': len(sentences),
            'words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'paragraph_count': text.count('\n\n') + 1,
            'bullet_point_count': text.count('•') + text.count('- ') + text.count('* ')
        }

    # -------- MÉTRICAS LÉXICAS -------- #
    def extraer_caracteristicas_lexicas(self, text):
        words = text.split()
        unique_words = set(words)

        return {
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_word_count': sum(1 for w in words if w.isupper() and len(w) > 1),
            'lexical_diversity': len(unique_words) / len(words) if words else 0
        }

    # -------- SENTIMIENTO -------- #
    def extraer_sentimiento_vader(self, text):
        scores = self.vader.polarity_scores(text)
        return {
            'vader_neg': scores['neg'],
            'vader_neu': scores['neu'],
            'vader_pos': scores['pos'],
            'vader_compound': scores['compound']
        }

    def extraer_sentimiento_textblob(self, text):
        try:
            blob = TextBlob(text)
            return {
                'textblob_polarity': blob.sentiment.polarity,
                'textblob_subjectivity': blob.sentiment.subjectivity
            }
        except Exception:
            return {'textblob_polarity': 0.0, 'textblob_subjectivity': 0.0}

    # -------- ESPECIFICIDAD DE ALIMENTOS -------- #
    def extraer_especificidad_alimentos(self, text):
        text_lower = text.lower()

        taste = ['sweet', 'salty', 'bitter', 'sour', 'umami', 'flavor', 'taste', 'spicy', 'bland']
        texture = ['crunchy', 'soft', 'chewy', 'tender', 'crispy', 'smooth', 'creamy', 'hard']
        quality = ['fresh', 'stale', 'rancid', 'expired', 'organic', 'natural', 'premium']

        score = sum(text_lower.count(w) for w in taste + texture + quality)

        return {'specificity_score': score}

    # -------- COMPARACIONES -------- #
    def extraer_comparaciones(self, text):
        text_lower = text.lower()
        comparison_words = ['than', 'better', 'worse', 'compared', 'versus', 'vs', 'instead', 'alternative', 'similar']
        return {'has_comparison': 1 if any(w in text_lower for w in comparison_words) else 0}

    # -------- EXPERIENCIA PERSONAL -------- #
    def extraer_experiencia_personal(self, text):
        text_lower = text.lower()
        personal = ['i ', 'my ', 'me ', 'we ', 'our ', "i've", "i'll"]
        time = ['days', 'weeks', 'months', 'years', 'always', 'daily', 'every', 'usually']

        return {
            'personal_experience_score': sum(text_lower.count(w) for w in personal + time)
        }

    # -------- PRECIO -------- #
    def extraer_menciones_precio(self, text):
        text_lower = text.lower()
        price_words = ['price', 'cost', 'expensive', 'cheap', 'worth', 'value', 'money', 'overpriced', 'affordable']
        return {'price_mention': 1 if any(w in text_lower for w in price_words) else 0}

    # -------- READABILITY -------- #
    def extraer_legibilidad(self, text):
        try:
            return {
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'gunning_fog': textstat.gunning_fog(text)
            }
        except:
            return {'flesch_reading_ease': 0, 'gunning_fog': 0}

    # -------- DIGITOS -------- #
    def extraer_caracteristicas_adicionales(self, text, score=None):
        return {'digit_ratio': sum(c.isdigit() for c in text) / len(text) if text else 0}

    # -------- MASTER FEATURE EXTRACTOR -------- #
    def extraer_todas_caracteristicas(self, text, score=None):
        features = {}
        features.update(self.extraer_longitud_texto(text))
        features.update(self.extraer_caracteristicas_lexicas(text))
        features.update(self.extraer_sentimiento_vader(text))
        features.update(self.extraer_sentimiento_textblob(text))
        features.update(self.extraer_caracteristicas_adicionales(text, score))
        features.update(self.extraer_especificidad_alimentos(text))
        features.update(self.extraer_comparaciones(text))
        features.update(self.extraer_experiencia_personal(text))
        features.update(self.extraer_menciones_precio(text))
        features.update(self.extraer_legibilidad(text))
        return features


def procesar_dataset(df, text_column='CleanText', score_column='Score'):
    """
    Procesa todo el dataset y extrae características NLP.
    """
    print("\n--- EXTRAYENDO CARACTERÍSTICAS NLP ---")
    print(f"Procesando {len(df)} reseñas desde columna '{text_column}'...")
    print(f"⚠️  Estimación de tiempo: ~{len(df)//10000} minutos para {len(df)} reseñas")

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


# ================================
# ANÁLISIS
# ================================

def analizar_correlaciones(df, target='IsHelpful'):
    print("\n--- CORRELACIONES CON LA VARIABLE OBJETIVO ---")
    num_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[num_cols].corrwith(df[target]).sort_values(ascending=False)

    print("\n🔝 Top 10 correlaciones positivas:")
    print(correlations.head(10))
    print("\n🔻 Top 10 correlaciones negativas:")
    print(correlations.tail(10))

    return correlations


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    print("=" * 60)
    print(" EXTRACCIÓN DE CARACTERÍSTICAS NLP ")
    print("=" * 60)

    data_path = os.path.join(SCRIPT_DIR, "..", "data", "amazon_reviews_prepared.csv")

    if not os.path.exists(data_path):
        print(f"\n❌ Error: No existe {data_path}")
        print("Ejecuta primero limpieza.py")
        sys.exit(1)

    print(f"\n✓ Cargando dataset desde: {data_path}")
    df = pd.read_csv(data_path)

    df_features = procesar_dataset(df, text_column='CleanText')

    output_path = os.path.join(SCRIPT_DIR, "..", "data", "amazon_reviews_with_features.csv")
    df_features.to_csv(output_path, index=False)

    print(f"\n✓ Dataset final guardado en: {output_path}")
    print("\n--- MUESTRA ---")
    print(df_features.head())

def obtener_estadisticas_features(df):
    """Imprime estadísticas básicas de las características extraídas."""
    print("\n--- ESTADÍSTICAS DE CARACTERÍSTICAS NLP ---")

    feature_columns = [
        'char_count', 'word_count', 'sentence_count',
        'vader_compound', 'textblob_polarity', 
        'lexical_diversity', 'flesch_reading_ease'
    ]

    for col in feature_columns:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  Media:   {df[col].mean():.3f}")
            print(f"  Mediana: {df[col].median():.3f}")
            print(f"  Std:     {df[col].std():.3f}")
