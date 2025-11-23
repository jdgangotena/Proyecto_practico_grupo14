
import requests
import sys

API_URL = "http://localhost:8000"

# A much longer, more detailed review
spanish_text = """
He probado docenas de marcas de té verde en los últimos 5 años, desde opciones de supermercado hasta importaciones japonesas premium, y puedo decir con confianza que este es el mejor en términos de relación calidad-precio.

Sabor y Aroma:
A diferencia del té verde de Lipton o Bigelow que a menudo sabe amargo o a "pasto quemado" si se deja reposar demasiado tiempo, este té mantiene un perfil de sabor dulce y umami incluso después de 5 minutos. Tiene notas claras de nuez y espinaca fresca, típicas de un buen Sencha. El aroma al abrir la bolsa es increíblemente fresco.

Calidad de la Hoja:
Lo que más me impresiona es que son hojas enteras enrolladas, no el "polvo" que encuentras en las bolsitas convencionales. Se expanden maravillosamente en el agua.

Efectos y Beneficios:
Lo tomo cada mañana antes de entrenar. Me da un impulso de energía sostenido durante 4-5 horas sin el bajón o la ansiedad que me provoca el café Starbucks o Nespresso. He notado una mejora en mi digestión y concentración.

Valor:
A $15 por 50 porciones, sale a $0.30 por taza. Considerando que puedes re-infusionar las hojas hasta 3 veces manteniendo el sabor (algo que no puedes hacer con marcas baratas), el valor real es excepcional. Es más caro que las marcas genéricas, pero obtienes 3 veces más rendimiento.

Veredicto:
Si buscas dejar el café o simplemente quieres un té verde auténtico sin pagar precios de ceremonia, este es el indicado. Definitivamente compraré la suscripción mensual.
"""

payload = {
    "text": spanish_text,
    "score": 5
}

print("\nSending High Quality Spanish review...")
try:
    response = requests.post(f"{API_URL}/reviews/predict_helpfulness", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print("\n--- Response ---")
        print(f"Probability: {data['is_helpful_probability']}")
        print(f"Is Helpful: {data['is_helpful']}")
        print(f"Confidence: {data['confidence']}")
        print(f"Translated Text: {data.get('translated_text', 'N/A')[:100]}...")
        
        print("\nFeatures:")
        for k, v in data['features'].items():
            print(f"  {k}: {v}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"Connection error: {e}")
