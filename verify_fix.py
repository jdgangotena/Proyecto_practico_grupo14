
import requests
import time
import sys

API_URL = "http://localhost:8000"

spanish_text = "Este té verde orgánico es absolutamente fantástico. Su sabor es fresco y limpio, con sutiles notas herbáceas que no resultan abrumadoras. Lo he estado bebiendo a diario durante los últimos tres meses y he notado una mayor energía sin el nerviosismo que me produce el café. La calidad es notablemente mejor que la de Lipton o Twinings: se ven las hojas enteras en lugar de polvo. Un poco caro, a $15 por caja, pero vale totalmente la pena por la calidad. Cada bolsita se puede remojar dos veces, lo que lo hace más económico. Lo recomiendo ampliamente para quienes buscan dejar de tomar café o ampliar su selección de tés."

payload = {
    "text": spanish_text,
    "score": 5
}

print("Waiting for API to start...")
for i in range(10):
    try:
        requests.get(f"{API_URL}/health")
        print("API is up!")
        break
    except:
        time.sleep(2)
else:
    print("API failed to start.")
    sys.exit(1)

print("\nSending Spanish review...")
response = requests.post(f"{API_URL}/reviews/predict_helpfulness", json=payload)

if response.status_code == 200:
    data = response.json()
    print("\n--- Response ---")
    print(f"Probability: {data['is_helpful_probability']}")
    print(f"Is Helpful: {data['is_helpful']}")
    print(f"Confidence: {data['confidence']}")
    print(f"Translated Text: {data.get('translated_text', 'N/A')}")
    print("Features:")
    for k, v in data['features'].items():
        print(f"  {k}: {v}")
    
    if data['is_helpful']:
        print(f"\nSUCCESS: Review predicted as HELPFUL! (Prob: {data['is_helpful_probability']})")
    else:
        print(f"\nFAILURE: Review predicted as NOT HELPFUL. (Prob: {data['is_helpful_probability']})")
        sys.exit(1)
else:
    print(f"\nError: {response.status_code}")
    print(response.text)
    sys.exit(1)
