"""
Pipeline Completo - Review Helpfulness Prediction
Ejecuta todo el pipeline desde la carga de datos hasta el entrenamiento del modelo.
"""

import os
import sys
import time
from datetime import datetime

# Añadir el directorio scripts al path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(SCRIPT_DIR, "scripts")
sys.path.append(SCRIPTS_DIR)


def print_section(title):
    """Imprime una sección con formato."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def print_step(step_num, total_steps, description):
    """Imprime el paso actual."""
    print(f"\n[{step_num}/{total_steps}] {description}")
    print("-" * 70)


def run_pipeline(nrows=50000, skip_training=False):
    """
    Ejecuta el pipeline completo.

    Args:
        nrows: Número de filas a procesar (None para todo el dataset)
        skip_training: Si True, salta el entrenamiento del modelo
    """
    start_time = time.time()

    print_section("PIPELINE DE PREDICCIÓN DE UTILIDAD DE RESEÑAS")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Procesando {nrows if nrows else 'todas las'} filas del dataset")

    try:
        # ===== PASO 1: CARGAR DATOS =====
        print_step(1, 4, "CARGANDO DATOS")

        from data_loader import cargar_datos, DATA_PATH

        if not os.path.exists(DATA_PATH):
            print(f"❌ ERROR: Dataset no encontrado en {DATA_PATH}")
            print("Por favor, descarga el dataset de Amazon Reviews desde:")
            print("https://www.kaggle.com/snap/amazon-fine-food-reviews")
            print("y colócalo en la carpeta 'data/' con el nombre 'Reviews.csv'")
            return False

        df = cargar_datos(DATA_PATH, nrows=nrows)

        if df is None:
            print("❌ ERROR: No se pudo cargar el dataset")
            return False

        print(f"✓ Dataset cargado: {len(df)} filas")

        # ===== PASO 2: LIMPIEZA Y PREPROCESAMIENTO =====
        print_step(2, 4, "LIMPIEZA Y PREPROCESAMIENTO")

        from limpieza import (
            calcular_tasa_utilidad,
            limpiar_texto_basico,
            preparar_dataset
        )

        # Calcular tasa de utilidad
        df = calcular_tasa_utilidad(df, umbral=0.7)

        if len(df) == 0:
            print("❌ ERROR: No hay datos después de filtrar por votos")
            return False

        # Limpiar texto
        df = limpiar_texto_basico(df)

        # Guardar dataset preparado
        output_path_cleaned = os.path.join(SCRIPT_DIR, "data", "amazon_reviews_prepared.csv")
        df_prepared = preparar_dataset(df, output_path_cleaned, limpieza_completa=False)

        print(f"✓ Limpieza completada: {len(df_prepared)} filas")

        # ===== PASO 3: EXTRACCIÓN DE CARACTERÍSTICAS NLP =====
        print_step(3, 4, "EXTRAYENDO CARACTERÍSTICAS NLP")

        from nlp_features import procesar_dataset, obtener_estadisticas_features

        # Extraer características
        df_con_features = procesar_dataset(df_prepared, text_column='CleanText', score_column='Score')

        # Estadísticas
        obtener_estadisticas_features(df_con_features)

        # Guardar dataset con características
        output_path_features = os.path.join(SCRIPT_DIR, "data", "amazon_reviews_with_features.csv")
        df_con_features.to_csv(output_path_features, index=False)

        print(f"✓ Características extraídas: {len(df_con_features.columns)} columnas")

        # ===== PASO 4: ENTRENAR MODELO =====
        if not skip_training:
            print_step(4, 4, "ENTRENANDO MODELO")

            from model_training import ReviewHelpfulnessModel, crear_graficos_evaluacion

            # Crear instancia del modelo
            model = ReviewHelpfulnessModel()

            # Preparar datos
            X_train, X_test, y_train, y_test = model.preparar_datos(df_con_features, target='IsHelpful')

            # Entrenar modelo
            model.entrenar(X_train, y_train)

            # Evaluar modelo
            metrics = model.evaluar(X_test, y_test)

            # Obtener importancia de características
            feature_importance = model.obtener_importancia_features()
            print("\nTop 10 características más importantes:")
            print(feature_importance.head(10))

            # Crear gráficos
            y_pred_proba = model.predecir(X_test)
            crear_graficos_evaluacion(y_test, y_pred_proba, feature_importance, save=True)

            # Guardar modelo
            model_path = model.guardar_modelo('review_helpfulness_model')

            print(f"✓ Modelo entrenado y guardado: {model_path}")
        else:
            print("\n[4/4] ENTRENAMIENTO OMITIDO (skip_training=True)")

        # ===== RESUMEN FINAL =====
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        print_section("PIPELINE COMPLETADO EXITOSAMENTE")

        print("📊 RESUMEN:")
        print(f"  • Filas procesadas: {len(df_con_features)}")
        print(f"  • Características extraídas: {len(df_con_features.columns)}")

        if not skip_training:
            print(f"\n📈 MÉTRICAS DEL MODELO:")
            for metric, value in metrics.items():
                print(f"  • {metric}: {value:.4f}")

        print(f"\n⏱️  Tiempo total: {minutes}m {seconds}s")
        print(f"✓ Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print("\n🚀 PRÓXIMOS PASOS:")
        print("  1. Iniciar la API:")
        print("     python api_app.py")
        print("\n  2. Iniciar el Dashboard:")
        print("     streamlit run dashboard.py")
        print("\n  3. Ver documentación de la API:")
        print("     http://localhost:8000/docs")

        return True

    except ImportError as e:
        print(f"\n❌ ERROR DE IMPORTACIÓN: {e}")
        print("Asegúrate de que todas las dependencias estén instaladas:")
        print("  pip install -r requirements.txt")
        return False

    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Función principal."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Pipeline completo de predicción de utilidad de reseñas'
    )
    parser.add_argument(
        '--nrows',
        type=int,
        default=50000,
        help='Número de filas a procesar (default: 50000, use 0 para todo el dataset)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Omitir el entrenamiento del modelo'
    )

    args = parser.parse_args()

    nrows = None if args.nrows == 0 else args.nrows

    success = run_pipeline(nrows=nrows, skip_training=args.skip_training)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
