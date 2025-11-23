
import sys
import os
from deep_translator import GoogleTranslator

# Add scripts to path
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

from nlp_features import NLPFeatureExtractor

spanish_text = "Este té verde orgánico es absolutamente fantástico. Su sabor es fresco y limpio, con sutiles notas herbáceas que no resultan abrumadoras. Lo he estado bebiendo a diario durante los últimos tres meses y he notado una mayor energía sin el nerviosismo que me produce el café. La calidad es notablemente mejor que la de Lipton o Twinings: se ven las hojas enteras en lugar de polvo. Un poco caro, a $15 por caja, pero vale totalmente la pena por la calidad. Cada bolsita se puede remojar dos veces, lo que lo hace más económico. Lo recomiendo ampliamente para quienes buscan dejar de tomar café o ampliar su selección de tés."

print(f"Original (Spanish): {spanish_text[:50]}...")

# Translate
try:
    translator = GoogleTranslator(source='auto', target='en')
    english_text = translator.translate(spanish_text)
    print(f"\nTranslated (English): {english_text}")
except Exception as e:
    print(f"Translation failed: {e}")
    sys.exit(1)

# Extract features from English text
extractor = NLPFeatureExtractor()
features = extractor.extraer_todas_caracteristicas(english_text, score=5)

print("\n--- Features for Translated Text ---")
for key, value in features.items():
    print(f"{key}: {value}")

print("\n--- Analysis ---")
print(f"VADER Compound: {features.get('vader_compound')} (Expected positive)")
print(f"Specificity Score: {features.get('specificity_score')} (Expected > 0)")
print(f"Has Comparison: {features.get('has_comparison')} (Expected 1)")
