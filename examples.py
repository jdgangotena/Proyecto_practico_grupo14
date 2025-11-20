"""
Ejemplos de Uso - Review Helpfulness API
Ejemplos prácticos de cómo usar la API de predicción de utilidad de reseñas.
"""

import requests
import json

# Configuración de la API
API_URL = "http://localhost:8000"


def verificar_api():
    """Verifica que la API esté funcionando."""
    print("="*60)
    print("VERIFICANDO CONEXIÓN CON LA API")
    print("="*60)

    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            print("\n✓ API conectada correctamente")
            print(f"  Estado: {data['status']}")
            print(f"  Modelo cargado: {data['model_loaded']}")
            if data['model_loaded']:
                print(f"  Características: {data['features_count']}")
            return True
        else:
            print("\n❌ Error al conectar con la API")
            return False
    except requests.exceptions.ConnectionError:
        print("\n❌ No se puede conectar a la API")
        print("Asegúrate de que la API esté ejecutándose:")
        print("  python api_app.py")
        return False


def ejemplo_1_resena_util():
    """Ejemplo 1: Reseña útil y detallada."""
    print("\n" + "="*60)
    print("EJEMPLO 1: RESEÑA ÚTIL")
    print("="*60)

    review = {
        "text": """This coffee maker is absolutely fantastic! I've been using it daily
        for three months now and it hasn't disappointed me once. The brewing temperature
        is perfect - not too hot, not too cold. The programmable timer is a lifesaver
        for early mornings. The carafe keeps coffee hot for hours without that burnt
        taste you get from some machines. The design is sleek and fits perfectly on
        my counter. Cleanup is easy - just rinse and go. Highly recommend for anyone
        looking for a reliable, quality coffee maker for daily use.""",
        "score": 5
    }

    print(f"\nTexto: {review['text'][:100]}...")
    print(f"Calificación: {review['score']} estrellas")

    try:
        response = requests.post(
            f"{API_URL}/reviews/predict_helpfulness",
            json=review,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ RESULTADOS:")
            print(f"  Probabilidad de utilidad: {data['is_helpful_probability']*100:.1f}%")
            print(f"  Clasificación: {'Útil' if data['is_helpful'] else 'Poco útil'}")
            print(f"  Confianza: {data['confidence']}")
            print(f"\n  Características destacadas:")
            print(f"    - Palabras: {data['features']['word_count']}")
            print(f"    - Oraciones: {data['features']['sentence_count']}")
            print(f"    - Sentimiento (VADER): {data['features']['vader_compound']:.3f}")
            print(f"\n  Sugerencias:")
            for i, suggestion in enumerate(data['suggestions'], 1):
                print(f"    {i}. {suggestion}")
        else:
            print(f"\n❌ Error {response.status_code}")

    except Exception as e:
        print(f"\n❌ Error: {e}")


def ejemplo_2_resena_corta():
    """Ejemplo 2: Reseña demasiado corta."""
    print("\n" + "="*60)
    print("EJEMPLO 2: RESEÑA CORTA")
    print("="*60)

    review = {
        "text": "Good product",
        "score": 4
    }

    print(f"\nTexto: '{review['text']}'")
    print(f"Calificación: {review['score']} estrellas")

    try:
        response = requests.post(
            f"{API_URL}/reviews/predict_helpfulness",
            json=review,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ RESULTADOS:")
            print(f"  Probabilidad de utilidad: {data['is_helpful_probability']*100:.1f}%")
            print(f"  Clasificación: {'Útil' if data['is_helpful'] else 'Poco útil'}")
            print(f"  Confianza: {data['confidence']}")
            print(f"\n  Sugerencias:")
            for i, suggestion in enumerate(data['suggestions'], 1):
                print(f"    {i}. {suggestion}")
        else:
            print(f"\n❌ Error {response.status_code}")

    except Exception as e:
        print(f"\n❌ Error: {e}")


def ejemplo_3_resena_negativa():
    """Ejemplo 3: Reseña negativa pero informativa."""
    print("\n" + "="*60)
    print("EJEMPLO 3: RESEÑA NEGATIVA INFORMATIVA")
    print("="*60)

    review = {
        "text": """I had high hopes for this blender, but unfortunately it didn't meet
        my expectations. The motor is quite weak and struggles with ice cubes. After
        just two weeks of use, it started making a grinding noise. The plastic container
        feels cheap and has already developed small cracks at the base. The blade assembly
        is difficult to clean properly. Customer service was unhelpful when I contacted
        them about the issues. Would not recommend this product.""",
        "score": 2
    }

    print(f"\nTexto: {review['text'][:100]}...")
    print(f"Calificación: {review['score']} estrellas")

    try:
        response = requests.post(
            f"{API_URL}/reviews/predict_helpfulness",
            json=review,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ RESULTADOS:")
            print(f"  Probabilidad de utilidad: {data['is_helpful_probability']*100:.1f}%")
            print(f"  Clasificación: {'Útil' if data['is_helpful'] else 'Poco útil'}")
            print(f"  Confianza: {data['confidence']}")
            print(f"\n  Características destacadas:")
            print(f"    - Palabras: {data['features']['word_count']}")
            print(f"    - Sentimiento (VADER): {data['features']['vader_compound']:.3f}")
            print(f"    - Polaridad (TextBlob): {data['features']['textblob_polarity']:.3f}")
            print(f"\n  Nota: Una reseña negativa puede ser útil si es informativa!")
        else:
            print(f"\n❌ Error {response.status_code}")

    except Exception as e:
        print(f"\n❌ Error: {e}")


def ejemplo_4_batch_predictions():
    """Ejemplo 4: Predicciones en lote."""
    print("\n" + "="*60)
    print("EJEMPLO 4: PREDICCIONES EN LOTE")
    print("="*60)

    reviews = [
        {"text": "Excellent! Works perfectly.", "score": 5},
        {"text": "Okay", "score": 3},
        {"text": "Terrible product. Broke after one use. Total waste of money.", "score": 1},
        {"text": "Great quality and fast shipping. The product does exactly what it promises.", "score": 5}
    ]

    print(f"\nProcesando {len(reviews)} reseñas...\n")

    results = []
    for i, review in enumerate(reviews, 1):
        try:
            response = requests.post(
                f"{API_URL}/reviews/predict_helpfulness",
                json=review,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                results.append({
                    "text": review["text"][:50] + "...",
                    "score": review["score"],
                    "probability": data["is_helpful_probability"],
                    "helpful": data["is_helpful"]
                })

                print(f"Reseña {i}:")
                print(f"  Texto: {review['text'][:50]}...")
                print(f"  Score: {review['score']} ⭐")
                print(f"  Utilidad: {data['is_helpful_probability']*100:.1f}% - {'✓ Útil' if data['is_helpful'] else '✗ Poco útil'}")
                print()

        except Exception as e:
            print(f"Error en reseña {i}: {e}")

    # Resumen
    if results:
        avg_probability = sum(r["probability"] for r in results) / len(results)
        helpful_count = sum(1 for r in results if r["helpful"])

        print("\n📊 RESUMEN:")
        print(f"  Total de reseñas: {len(results)}")
        print(f"  Reseñas útiles: {helpful_count} ({helpful_count/len(results)*100:.1f}%)")
        print(f"  Probabilidad promedio: {avg_probability*100:.1f}%")


def ejemplo_5_comparar_calificaciones():
    """Ejemplo 5: Comparar misma reseña con diferentes calificaciones."""
    print("\n" + "="*60)
    print("EJEMPLO 5: IMPACTO DE LA CALIFICACIÓN")
    print("="*60)

    text = """This product has good features and decent build quality.
    The price is reasonable for what you get. Setup was straightforward."""

    print(f"\nTexto fijo: {text}\n")
    print("Probando con diferentes calificaciones...\n")

    for score in range(1, 6):
        review = {"text": text, "score": score}

        try:
            response = requests.post(
                f"{API_URL}/reviews/predict_helpfulness",
                json=review,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                print(f"Calificación {score} ⭐: {data['is_helpful_probability']*100:.1f}% de utilidad")

        except Exception as e:
            print(f"Error con score {score}: {e}")


def obtener_info_modelo():
    """Obtiene información sobre el modelo."""
    print("\n" + "="*60)
    print("INFORMACIÓN DEL MODELO")
    print("="*60)

    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)

        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ Información del modelo:")
            print(f"  Ruta: {data['model_path']}")
            print(f"  Características: {data['features_count']}")
            print(f"  Timestamp: {data['timestamp']}")

            if data.get('metrics'):
                print(f"\n  Métricas de evaluación:")
                for metric, value in data['metrics'].items():
                    print(f"    - {metric}: {value:.4f}")

            print(f"\n  Características usadas:")
            for i, feature in enumerate(data['feature_columns'][:10], 1):
                print(f"    {i}. {feature}")
            if len(data['feature_columns']) > 10:
                print(f"    ... y {len(data['feature_columns']) - 10} más")

        else:
            print(f"\n❌ Error {response.status_code}")

    except Exception as e:
        print(f"\n❌ Error: {e}")


def main():
    """Función principal."""
    print("\n🔍 EJEMPLOS DE USO - REVIEW HELPFULNESS API")
    print("="*60)
    print("\nAsegúrate de que la API esté ejecutándose:")
    print("  python api_app.py\n")

    # Verificar conexión
    if not verificar_api():
        return

    # Ejecutar ejemplos
    ejemplos = [
        ("1", "Reseña útil y detallada", ejemplo_1_resena_util),
        ("2", "Reseña corta", ejemplo_2_resena_corta),
        ("3", "Reseña negativa informativa", ejemplo_3_resena_negativa),
        ("4", "Predicciones en lote", ejemplo_4_batch_predictions),
        ("5", "Impacto de la calificación", ejemplo_5_comparar_calificaciones),
        ("6", "Información del modelo", obtener_info_modelo),
    ]

    print("\n\nEJEMPLOS DISPONIBLES:")
    for num, desc, _ in ejemplos:
        print(f"  {num}. {desc}")
    print("  0. Ejecutar todos")
    print("  q. Salir")

    while True:
        opcion = input("\nSelecciona un ejemplo (0-6, q para salir): ").strip().lower()

        if opcion == 'q':
            print("\n¡Hasta luego!")
            break
        elif opcion == '0':
            for _, _, func in ejemplos:
                func()
                input("\nPresiona Enter para continuar...")
            break
        elif opcion in [e[0] for e in ejemplos]:
            func = next(e[2] for e in ejemplos if e[0] == opcion)
            func()
        else:
            print("Opción inválida. Intenta de nuevo.")


if __name__ == "__main__":
    main()
