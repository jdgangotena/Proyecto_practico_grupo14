"""
Limpieza y Preprocesamiento de Datos - Amazon Reviews
Limpia el texto de las reseñas y calcula la tasa de utilidad para clasificación.
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import sys

# Añadir el directorio scripts al path para importar data_loader
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from data_loader import cargar_datos, DATA_PATH

# Descargar recursos necesarios de NLTK (solo la primera vez)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


def calcular_tasa_utilidad(df, umbral=0.7):
    """
    Calcula la tasa de utilidad y crea una etiqueta binaria.

    Args:
        df: DataFrame con columnas HelpfulnessNumerator y HelpfulnessDenominator
        umbral: Umbral para clasificar como útil (default: 0.7)

    Returns:
        DataFrame con columnas adicionales: HelpfulnessRate y IsHelpful
    """
    print("\n--- CALCULANDO TASA DE UTILIDAD ---")

    # Filtrar reseñas con al menos 1 voto
    df_filtered = df[df['HelpfulnessDenominator'] >= 1].copy()
    print(f"Reseñas con al menos 1 voto: {len(df_filtered)} de {len(df)} ({len(df_filtered)/len(df)*100:.1f}%)")

    # Calcular tasa de utilidad
    df_filtered['HelpfulnessRate'] = (
        df_filtered['HelpfulnessNumerator'] / df_filtered['HelpfulnessDenominator']
    )

    # Crear etiqueta binaria: útil (1) si tasa >= umbral
    df_filtered['IsHelpful'] = (df_filtered['HelpfulnessRate'] >= umbral).astype(int)

    # Estadísticas
    print(f"\nDistribución de etiquetas:")
    print(df_filtered['IsHelpful'].value_counts())
    print(f"\nPorcentaje de reseñas útiles: {df_filtered['IsHelpful'].mean()*100:.1f}%")
    print(f"Tasa de utilidad promedio: {df_filtered['HelpfulnessRate'].mean():.2%}")

    return df_filtered


def limpiar_texto_basico(df):
    """
    Realiza limpieza básica del texto sin remover stopwords ni lematizar.
    Útil para preservar más información para análisis de sentimiento.

    Args:
        df: DataFrame con columnas Summary y Text

    Returns:
        DataFrame con columna adicional: CleanText
    """
    print("\n--- LIMPIEZA BÁSICA DE TEXTO ---")

    # Unificar texto (Resumen + Texto completo)
    df["FullReview"] = df["Summary"].fillna("").astype(str) + " " + df["Text"].fillna("").astype(str)

    def clean_text_basic(text):
        """Limpieza básica: lowercase, URLs, caracteres especiales"""
        # Convertir a minúsculas
        text = text.lower()

        # Eliminar URLs
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

        # Eliminar caracteres especiales pero mantener puntuación básica
        text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)

        # Eliminar espacios extra
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    df["CleanText"] = df["FullReview"].apply(clean_text_basic)

    # Estadísticas
    df['text_length'] = df['CleanText'].str.len()
    print(f"Longitud promedio del texto limpio: {df['text_length'].mean():.0f} caracteres")
    print(f"Longitud mediana: {df['text_length'].median():.0f} caracteres")

    return df


def limpiar_texto_completo(text):
    """
    Limpieza completa de texto: lowercase, stopwords, lematización.

    Args:
        text: Texto a limpiar

    Returns:
        Texto limpio
    """
    # Convertir a minúsculas
    text = text.lower()

    # Eliminar URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

    # Eliminar puntuación y caracteres especiales
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Eliminar números
    text = re.sub(r'\d+', '', text)

    # Eliminar espacios extra
    text = re.sub(r'\s+', ' ', text).strip()

    # Remover stopwords
    stop_words = set(stopwords.words("english"))
    tokens = nltk.word_tokenize(text)
    filtered_words = [w for w in tokens if w and w not in stop_words]

    # Lematización
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in filtered_words]

    return " ".join(lemmas)


def aplicar_limpieza_completa(df):
    """
    Aplica limpieza completa con stopwords y lematización.

    Args:
        df: DataFrame con columna CleanText

    Returns:
        DataFrame con columna adicional: ProcessedText
    """
    print("\n--- LIMPIEZA COMPLETA DE TEXTO ---")
    print("Aplicando stopwords removal y lematización...")

    df["ProcessedText"] = df["CleanText"].apply(limpiar_texto_completo)

    print("✓ Limpieza completa finalizada")

    return df


def preparar_dataset(df, output_path, limpieza_completa=False):
    """
    Prepara el dataset final para entrenamiento.

    Args:
        df: DataFrame procesado
        output_path: Ruta para guardar el CSV
        limpieza_completa: Si True, incluye ProcessedText

    Returns:
        DataFrame preparado
    """
    print("\n--- PREPARANDO DATASET FINAL ---")

    # Seleccionar columnas relevantes
    columnas = [
        'ProductId', 'UserId', 'Score', 'Time',
        'HelpfulnessNumerator', 'HelpfulnessDenominator',
        'HelpfulnessRate', 'IsHelpful',
        'FullReview', 'CleanText'
    ]

    if limpieza_completa and 'ProcessedText' in df.columns:
        columnas.append('ProcessedText')

    df_prepared = df[columnas].copy()

    # Guardar
    df_prepared.to_csv(output_path, index=False)
    print(f"✓ Dataset guardado en: {output_path}")
    print(f"Dimensiones: {df_prepared.shape[0]} filas × {df_prepared.shape[1]} columnas")

    return df_prepared


if __name__ == "__main__":
    print("="*60)
    print("PIPELINE DE LIMPIEZA Y PREPROCESAMIENTO")
    print("="*60)

    # 1. Cargar datos
    df = cargar_datos(DATA_PATH, nrows=50000)  # Cargar subset para pruebas

    if df is None:
        print("Error al cargar datos. Abortando.")
        sys.exit(1)

    # 2. Calcular tasa de utilidad
    df = calcular_tasa_utilidad(df, umbral=0.7)

    # 3. Limpieza básica de texto
    df = limpiar_texto_basico(df)

    # 4. (Opcional) Limpieza completa
    # df = aplicar_limpieza_completa(df)

    # 5. Preparar y guardar dataset
    output_path = os.path.join(SCRIPT_DIR, "..", "data", "amazon_reviews_prepared.csv")
    df_prepared = preparar_dataset(df, output_path, limpieza_completa=False)

    print("\n--- MUESTRA DEL DATASET PREPARADO ---")
    print(df_prepared[['Score', 'HelpfulnessRate', 'IsHelpful', 'CleanText']].head())

    print("\n✓ Preprocesamiento completado exitosamente")
