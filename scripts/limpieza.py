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


def calcular_tasa_utilidad(df, umbral=0.7, min_votes=None):
    """
    Calcula la tasa de utilidad y crea una etiqueta binaria.

    CRÍTICO: No filtra reseñas con 0 votos por defecto para evitar sesgo.
    Las reseñas sin votos reciben HelpfulnessRate = 0 (asumiendo no útil).

    Args:
        df: DataFrame con columnas HelpfulnessNumerator y HelpfulnessDenominator
        umbral: Umbral para clasificar como útil (default: 0.7)
        min_votes: Si se especifica, filtra reseñas con menos de N votos
                   (None = incluir todas, incluso con 0 votos)

    Returns:
        DataFrame con columnas adicionales: HelpfulnessRate y IsHelpful
    """
    print("\n--- CALCULANDO TASA DE UTILIDAD ---")

    df_processed = df.copy()

    # CRÍTICO: Manejar reseñas con 0 votos sin filtrarlas (asignar rate = 0)
    # Esto evita pérdida masiva de datos y sesgo
    df_processed['HelpfulnessRate'] = 0.0  # Default para 0 votos

    # Calcular tasa solo donde hay votos
    has_votes = df_processed['HelpfulnessDenominator'] > 0
    df_processed.loc[has_votes, 'HelpfulnessRate'] = (
        df_processed.loc[has_votes, 'HelpfulnessNumerator'] /
        df_processed.loc[has_votes, 'HelpfulnessDenominator']
    )

    # Crear etiqueta binaria: útil (1) si tasa >= umbral
    df_processed['IsHelpful'] = (df_processed['HelpfulnessRate'] >= umbral).astype(int)

    # Filtrado opcional por mínimo de votos (si se especifica)
    if min_votes is not None and min_votes > 0:
        print(f"\n⚠️  Filtrado opcional activo: min_votes = {min_votes}")
        df_before = len(df_processed)
        df_processed = df_processed[df_processed['HelpfulnessDenominator'] >= min_votes].copy()
        print(f"Reseñas filtradas: {df_before - len(df_processed)} ({(df_before - len(df_processed))/df_before*100:.1f}%)")

    # Estadísticas
    print(f"\nReseñas totales procesadas: {len(df_processed)}")
    print(f"Reseñas con votos: {has_votes.sum()} ({has_votes.sum()/len(df)*100:.1f}%)")
    print(f"Reseñas sin votos: {(~has_votes).sum()} ({(~has_votes).sum()/len(df)*100:.1f}%)")

    print(f"\nDistribución de etiquetas:")
    print(df_processed['IsHelpful'].value_counts())
    print(f"\nPorcentaje de reseñas útiles: {df_processed['IsHelpful'].mean()*100:.1f}%")

    if has_votes.sum() > 0:
        print(f"Tasa de utilidad promedio (con votos): {df_processed.loc[has_votes, 'HelpfulnessRate'].mean():.2%}")

    return df_processed


def limpiar_texto_basico(df):
    """
    Realiza limpieza básica del texto sin remover stopwords ni lematizar.
    Útil para preservar más información para análisis de sentimiento.

    PIPELINE DE COLUMNAS:
    - FullReview: Summary + Text concatenados (texto original completo)
    - CleanText: Limpieza básica (lowercase, URLs, chars especiales)
                 → ESTA COLUMNA SE USA PARA FEATURES NLP
    - ProcessedText: (Opcional) Limpieza completa con stopwords + lematización
                     → NO se recomienda para sentiment analysis

    Args:
        df: DataFrame con columnas Summary y Text

    Returns:
        DataFrame con columnas adicionales: FullReview, CleanText
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


def aplicar_limpieza_completa(df, batch_size=1000):
    """
    Aplica limpieza completa con stopwords y lematización.

    MEJORA: Procesa en batches para evitar problemas de memoria en datasets grandes.

    Args:
        df: DataFrame con columna CleanText
        batch_size: Tamaño del batch para procesamiento (default: 1000)

    Returns:
        DataFrame con columna adicional: ProcessedText
    """
    print("\n--- LIMPIEZA COMPLETA DE TEXTO ---")
    print(f"Aplicando stopwords removal y lematización en batches de {batch_size}...")

    total_rows = len(df)
    processed_texts = []

    # Procesar en batches para mejor rendimiento
    for i in range(0, total_rows, batch_size):
        batch_end = min(i + batch_size, total_rows)
        batch = df["CleanText"].iloc[i:batch_end]

        # Aplicar limpieza al batch
        batch_processed = batch.apply(limpiar_texto_completo)
        processed_texts.extend(batch_processed.tolist())

        # Mostrar progreso
        print(f"  Procesadas {batch_end}/{total_rows} reseñas ({batch_end/total_rows*100:.1f}%)")

    df["ProcessedText"] = processed_texts

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

    # ALTA: Asegurar que el directorio de salida existe
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Solo si hay un directorio (no es ruta relativa simple)
        os.makedirs(output_dir, exist_ok=True)

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
    # Por defecto: incluye TODAS las reseñas (incluso con 0 votos) para evitar sesgo
    # Opcional: usar min_votes para filtrar (ej: min_votes=5 para solo reseñas con 5+ votos)
    df = calcular_tasa_utilidad(df, umbral=0.7, min_votes=None)
    
    # 🔥 Filtrar reseñas con tasa 0 (NO útiles o sin votos TODO: Borrar)
    df = df[df["HelpfulnessRate"] > 0].copy()
    print(f"Reseñas restantes tras filtrar HelpfulnessRate=0: {len(df)}")

    # 3. Limpieza básica de texto
    df = limpiar_texto_basico(df)

    # 4. (Opcional) Limpieza completa con lematización
    # Procesamiento por batches para mejor rendimiento
    # df = aplicar_limpieza_completa(df, batch_size=1000)

    # 5. Preparar y guardar dataset
    output_path = os.path.join(SCRIPT_DIR, "..", "data", "amazon_reviews_prepared.csv")
    df_prepared = preparar_dataset(df, output_path, limpieza_completa=False)

    print("\n--- MUESTRA DEL DATASET PREPARADO ---")
    print(df_prepared[['Score', 'HelpfulnessRate', 'IsHelpful', 'CleanText']].head())

    print("\n✓ Preprocesamiento completado exitosamente")
