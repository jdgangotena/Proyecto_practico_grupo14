"""
Data Loader - Amazon Reviews Dataset
Carga y valida el dataset de reseñas de productos de Amazon.
"""

import pandas as pd
import os
import urllib.request
from pathlib import Path

# Configuración de rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
DATA_PATH = os.path.join(DATA_DIR, "Reviews.csv")

# URL del dataset (Amazon Fine Food Reviews)
DATASET_URL = "https://snap.stanford.edu/data/finefoods.txt.gz"


def descargar_dataset(url=DATASET_URL, output_dir=DATA_DIR):
    """
    Descarga el dataset de Amazon Reviews si no existe localmente.

    Args:
        url: URL del dataset
        output_dir: Directorio de salida
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, "Reviews.csv")

    if os.path.exists(output_path):
        print(f"El dataset ya existe en: {output_path}")
        return output_path

    print(f"Descargando dataset desde {url}...")
    print("NOTA: Para este proyecto, asegúrate de tener el archivo Reviews.csv en la carpeta 'data'")
    print("Puedes descargarlo desde: https://www.kaggle.com/snap/amazon-fine-food-reviews")

    return output_path


def validar_columnas(df):
    """
    Valida que el dataset contenga las columnas necesarias.

    Args:
        df: DataFrame a validar

    Returns:
        bool: True si todas las columnas están presentes
    """
    columnas_requeridas = [
        'Id', 'ProductId', 'UserId', 'ProfileName',
        'HelpfulnessNumerator', 'HelpfulnessDenominator',
        'Score', 'Time', 'Summary', 'Text'
    ]

    columnas_faltantes = set(columnas_requeridas) - set(df.columns)

    if columnas_faltantes:
        print(f"⚠️ Advertencia: Faltan las siguientes columnas: {columnas_faltantes}")
        return False

    print("✓ Todas las columnas requeridas están presentes")
    return True


def cargar_datos(path=DATA_PATH, nrows=None):
    """
    Carga el dataset de reseñas desde un archivo CSV.

    Args:
        path: Ruta al archivo CSV
        nrows: Número de filas a cargar (None para cargar todas)

    Returns:
        pd.DataFrame: DataFrame con los datos cargados, o None si hay error
    """
    print(f"Cargando datos desde {path}...")

    try:
        # Cargar el dataset
        df = pd.read_csv(path, nrows=nrows)
        print(f"✓ Datos cargados exitosamente: {len(df)} filas, {len(df.columns)} columnas")

        # Validar columnas
        validar_columnas(df)

        # Mostrar información básica
        print(f"\nRango de fechas: {pd.to_datetime(df['Time'], unit='s').min()} a {pd.to_datetime(df['Time'], unit='s').max()}")
        print(f"Productos únicos: {df['ProductId'].nunique()}")
        print(f"Usuarios únicos: {df['UserId'].nunique()}")

        return df

    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo en {path}")
        print(f"Asegúrate de que el archivo existe en la carpeta 'data'")
        print(f"Puedes descargarlo desde: https://www.kaggle.com/snap/amazon-fine-food-reviews")
        return None

    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return None


def obtener_estadisticas_basicas(df):
    """
    Calcula y muestra estadísticas básicas del dataset.

    Args:
        df: DataFrame con los datos
    """
    print("\n" + "="*60)
    print("ESTADÍSTICAS BÁSICAS DEL DATASET")
    print("="*60)

    print(f"\nDimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")

    print(f"\nValores nulos por columna:")
    nulos = df.isnull().sum()
    for col, count in nulos.items():
        if count > 0:
            pct = (count / len(df)) * 100
            print(f"  {col}: {count} ({pct:.2f}%)")

    print(f"\nDistribución de calificaciones (Score):")
    print(df['Score'].value_counts().sort_index())

    print(f"\nEstadísticas de utilidad:")
    print(f"  HelpfulnessNumerator - Media: {df['HelpfulnessNumerator'].mean():.2f}")
    print(f"  HelpfulnessDenominator - Media: {df['HelpfulnessDenominator'].mean():.2f}")

    # Calcular tasa de utilidad
    df_temp = df[df['HelpfulnessDenominator'] > 0].copy()
    df_temp['HelpfulnessRate'] = df_temp['HelpfulnessNumerator'] / df_temp['HelpfulnessDenominator']
    print(f"  Tasa de utilidad promedio: {df_temp['HelpfulnessRate'].mean():.2%}")


if __name__ == "__main__":
    print(f"Ejecutando script desde: {os.path.abspath(__file__)}\n")

    # Verificar si existe el dataset, si no, informar sobre descarga
    if not os.path.exists(DATA_PATH):
        descargar_dataset()

    # Cargar datos (limitar a 50000 filas para pruebas rápidas, quitar nrows para cargar todo)
    df = cargar_datos(DATA_PATH, nrows=50000)

    if df is not None:
        print("\n--- PRIMERAS FILAS ---")
        print(df.head())

        print("\n--- INFORMACIÓN DEL DATAFRAME ---")
        df.info(show_counts=True)

        obtener_estadisticas_basicas(df)
