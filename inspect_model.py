
import sys
import os
import pandas as pd
import pickle
import json

# Add scripts to path
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

MODEL_PATH = "models/review_helpfulness_model_latest.pkl"
METADATA_PATH = "models/review_helpfulness_model_latest_metadata.json"

print(f"Loading model from {MODEL_PATH}...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)
    feature_columns = metadata.get('feature_columns', [])

print(f"Model loaded. Features: {len(feature_columns)}")

# Get feature importance
importances = model.feature_importance(importance_type='gain')
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n--- TOP 20 FEATURES ---")
print(feature_importance.head(20))
