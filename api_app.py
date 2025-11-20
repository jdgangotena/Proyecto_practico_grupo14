"""
API FastAPI - Review Helpfulness Prediction
API para predecir la utilidad de reseñas de productos.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import os
import sys
import pickle
import json

# Obtener el directorio actual del archivo
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Añadir el directorio raíz al path (para importar desde scripts)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Importar el extractor de características
try:
    from scripts.nlp_features import NLPFeatureExtractor
    print("✓ NLPFeatureExtractor importado correctamente")
except ImportError as e:
    print(f"⚠️ Error al importar NLPFeatureExtractor: {e}")
    print(f"⚠️ Asegúrate de que existe scripts/__init__.py y scripts/nlp_features.py")
    NLPFeatureExtractor = None

# Configuración
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "review_helpfulness_model_latest.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "review_helpfulness_model_latest_metadata.json")

# Inicializar FastAPI
app = FastAPI(
    title="Review Helpfulness Prediction API",
    description="API para predecir la utilidad de reseñas de productos usando Machine Learning",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales para el modelo
model = None
feature_columns = None
feature_extractor = None


# Modelos de datos
class ReviewInput(BaseModel):
    """Modelo de entrada para una reseña."""
    text: str = Field(..., description="Texto de la reseña", min_length=10)
    score: int = Field(..., description="Calificación en estrellas (1-5)", ge=1, le=5)

    class Config:
        schema_extra = {
            "example": {
                "text": "This product is amazing! It works exactly as described and the quality is excellent. I highly recommend it to anyone looking for a reliable solution.",
                "score": 5
            }
        }


class PredictionResponse(BaseModel):
    """Modelo de respuesta de predicción."""
    is_helpful_probability: float = Field(..., description="Probabilidad de que la reseña sea útil (0-1)")
    is_helpful: bool = Field(..., description="Predicción binaria: True si es útil, False si no")
    confidence: str = Field(..., description="Nivel de confianza: 'high', 'medium', 'low'")
    features: Dict[str, float] = Field(..., description="Características extraídas de la reseña")
    suggestions: List[str] = Field(..., description="Sugerencias para mejorar la reseña")


class HealthResponse(BaseModel):
    """Modelo de respuesta de health check."""
    status: str
    model_loaded: bool
    model_path: Optional[str] = None
    features_count: Optional[int] = None


# Funciones auxiliares
def cargar_modelo():
    """Carga el modelo y sus metadatos."""
    global model, feature_columns, feature_extractor

    if NLPFeatureExtractor is None:
        raise ImportError(
            "NLPFeatureExtractor no pudo ser importado. "
            "Verifica que existe scripts/__init__.py y scripts/nlp_features.py"
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modelo no encontrado en {MODEL_PATH}. "
            "Ejecuta model_training.py primero para entrenar el modelo."
        )

    # Cargar modelo
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    # Cargar metadatos
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
            feature_columns = metadata.get('feature_columns', [])
    else:
        raise FileNotFoundError(f"Metadatos no encontrados en {METADATA_PATH}")

    # Inicializar extractor de características
    feature_extractor = NLPFeatureExtractor()

    print(f"✓ Modelo cargado desde: {MODEL_PATH}")
    print(f"✓ Características: {len(feature_columns)}")


def generar_sugerencias(features: Dict[str, float], probability: float) -> List[str]:
    """
    Genera sugerencias para mejorar la reseña basándose en las características.

    Args:
        features: Características extraídas
        probability: Probabilidad de ser útil

    Returns:
        Lista de sugerencias
    """
    sugerencias = []

    # Longitud del texto
    word_count = features.get('word_count', 0)
    if word_count < 20:
        sugerencias.append("Tu reseña es muy corta. Añade más detalles sobre tu experiencia con el producto.")
    elif word_count < 50:
        sugerencias.append("Considera añadir más información específica sobre las características del producto.")

    # Número de oraciones
    sentence_count = features.get('sentence_count', 0)
    if sentence_count < 3:
        sugerencias.append("Estructura tu reseña en varios puntos para hacerla más clara y fácil de leer.")

    # Sentimiento
    vader_compound = features.get('vader_compound', 0)
    textblob_polarity = features.get('textblob_polarity', 0)

    if abs(vader_compound) < 0.2 and abs(textblob_polarity) < 0.2:
        sugerencias.append("Tu reseña parece neutral. Expresa claramente si recomiendas el producto y por qué.")

    # Diversidad léxica
    lexical_diversity = features.get('lexical_diversity', 0)
    if lexical_diversity < 0.5:
        sugerencias.append("Usa un vocabulario más variado para hacer tu reseña más interesante.")

    # Signos de interrogación/exclamación
    question_count = features.get('question_count', 0)
    exclamation_count = features.get('exclamation_count', 0)

    if question_count > 3:
        sugerencias.append("Evita hacer demasiadas preguntas. Proporciona respuestas y opiniones claras.")

    if exclamation_count > 5:
        sugerencias.append("Reduce el uso excesivo de exclamaciones para dar un tono más profesional.")

    # Si la probabilidad ya es alta
    if probability > 0.7 and not sugerencias:
        sugerencias.append("¡Excelente reseña! Es informativa y probablemente será útil para otros usuarios.")
    elif probability > 0.5 and not sugerencias:
        sugerencias.append("Tu reseña es buena, pero podría beneficiarse de más detalles específicos.")
    elif not sugerencias:
        sugerencias.append("Intenta ser más específico y detallado sobre tu experiencia con el producto.")

    return sugerencias


def clasificar_confianza(probability: float) -> str:
    """
    Clasifica el nivel de confianza de la predicción.

    Args:
        probability: Probabilidad predicha

    Returns:
        Nivel de confianza: 'high', 'medium', 'low'
    """
    if probability >= 0.7 or probability <= 0.3:
        return "high"
    elif probability >= 0.55 or probability <= 0.45:
        return "medium"
    else:
        return "low"


# Eventos de la aplicación
@app.on_event("startup")
async def startup_event():
    """Carga el modelo al iniciar la aplicación."""
    try:
        cargar_modelo()
        print("✓ API iniciada correctamente")
    except Exception as e:
        print(f"⚠️ Advertencia: No se pudo cargar el modelo: {e}")
        print("La API iniciará pero /reviews/predict_helpfulness no funcionará hasta que se entrene un modelo.")


# Endpoints
@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz."""
    return {
        "message": "Review Helpfulness Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/reviews/predict_helpfulness",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Verifica el estado de la API y el modelo."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_path=MODEL_PATH if model is not None else None,
        features_count=len(feature_columns) if feature_columns else None
    )


@app.post("/reviews/predict_helpfulness", response_model=PredictionResponse, tags=["Predictions"])
async def predict_helpfulness(review: ReviewInput):
    """
    Predice la utilidad de una reseña.

    Args:
        review: Reseña a evaluar

    Returns:
        Predicción de utilidad con sugerencias
    """
    if model is None or feature_extractor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Por favor entrena el modelo primero ejecutando model_training.py"
        )

    try:
        # Extraer características
        features = feature_extractor.extraer_todas_caracteristicas(review.text, review.score)

        # Preparar datos para predicción
        feature_values = [features.get(col, 0) for col in feature_columns]

        # Realizar predicción
        probability = float(model.predict([feature_values])[0])

        # Clasificar
        is_helpful = probability >= 0.5
        confidence = clasificar_confianza(probability)

        # Generar sugerencias
        suggestions = generar_sugerencias(features, probability)

        return PredictionResponse(
            is_helpful_probability=round(probability, 4),
            is_helpful=is_helpful,
            confidence=confidence,
            features=features,
            suggestions=suggestions
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la reseña: {str(e)}"
        )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Obtiene información sobre el modelo cargado."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible"
        )

    # Leer metadatos
    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)

    return {
        "model_path": MODEL_PATH,
        "features_count": len(feature_columns),
        "feature_columns": feature_columns,
        "metrics": metadata.get('metrics', {}),
        "timestamp": metadata.get('timestamp', 'unknown')
    }


# Punto de entrada para desarrollo
if __name__ == "__main__":
    import uvicorn

    print("="*60)
    print("INICIANDO API - REVIEW HELPFULNESS PREDICTION")
    print("="*60)
    print("\nDocumentación disponible en: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/health")
    print("\nPresiona Ctrl+C para detener el servidor\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)