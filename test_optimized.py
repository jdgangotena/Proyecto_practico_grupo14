
import requests

API_URL = "http://localhost:8000"

# Optimized review with top keywords: ginger, perfect, years, the best, use, like
spanish_text = "He estado bebiendo este té de jengibre durante años y es simplemente el mejor. El sabor es perfecto: fuerte pero no abrumador. Lo uso todas las mañanas para ayudar con la digestión. A diferencia de otras marcas, esta tiene un toque real de jengibre. Es un poco caro pero vale la pena. Lo recomiendo encarecidamente."

payload = {
    "text": spanish_text,
    "score": 5
}

print("\nSending Optimized Spanish review...")
try:
    response = requests.post(f"{API_URL}/reviews/predict_helpfulness", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print("\n--- Response ---")
        print(f"Probability: {data['is_helpful_probability']}")
        print(f"Is Helpful: {data['is_helpful']}")
        print(f"Translated Text: {data.get('translated_text', 'N/A')}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"Connection error: {e}")
