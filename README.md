# 🔍 Sistema de Predicción de Utilidad de Reseñas

Sistema de Machine Learning y NLP para predecir la utilidad de reseñas de productos usando clasificación supervisada con LightGBM.

## 📋 Descripción del Proyecto

Este proyecto construye un **Asistente de Reseñas** que predice si una reseña será considerada útil por otros usuarios, basándose en características extraídas del texto mediante técnicas de NLP (Procesamiento de Lenguaje Natural).

### Componentes Principales

1. **Pipeline de Datos**: Carga, limpieza y preprocesamiento de reseñas de Amazon
2. **Ingeniería de Características NLP**: Extracción de características (longitud, sentimiento, estructura, etc.)
3. **Modelo de Clasificación**: LightGBM para predecir utilidad binaria
4. **API REST**: FastAPI para servir predicciones
5. **Dashboard Interactivo**: Interfaz web para escribir reseñas y obtener feedback en tiempo real

## 🎯 Objetivo

Predecir la "puntuación de utilidad" de una reseña calculando características de calidad del texto y entrenando un modelo que aprenda la relación entre estas características y la utilidad percibida por usuarios.

## 🗂️ Estructura del Proyecto por carpetas

```
opiniones_ecommners-1/
├── scripts/
│   ├── data_loader.py         # Carga y validación de datos
│   ├── limpieza.py            # Limpieza y cálculo de tasa de utilidad
│   ├── nlp_features.py        # Extracción de características NLP
│   └── model_training.py      # Entrenamiento del modelo LightGBM
├── api_app.py                 # API FastAPI
├── dashboard.py               # Dashboard interactivo con Dash
├── requirements.txt           # Dependencias del proyecto
├── data/                      # Datos (no incluido en repo)
│   ├── Reviews.csv
│   ├── amazon_reviews_prepared.csv
│   └── amazon_reviews_with_features.csv
├── models/                    # Modelos entrenados
│   └── review_helpfulness_model_latest.pkl
└── plots/                     # Gráficos generados
    ├── roc_curve.html
    ├── feature_importance.html
    └── probability_distribution.html
```

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
cd opiniones_ecommners-1
```

### 2. Crear entorno virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Instalar dependencias requirements

```bash
pip install -r requirements.txt
```

### 4. Descargar dataset

Descarga el dataset **Amazon Fine Food Reviews** desde Kaggle:
- URL: https://www.kaggle.com/snap/amazon-fine-food-reviews
- Coloca el archivo `Reviews.csv` en la carpeta `data/`

## 📊 Pipeline de Ejecución

### Paso 1: Cargar y Explorar Datos

```bash
cd scripts
python data_loader.py
```

**Funcionalidades:**
- Carga el dataset de reseñas
- Valida columnas requeridas
- Muestra estadísticas básicas
- Calcula tasa de utilidad promedio

### Paso 2: Limpieza y Preprocesamiento

```bash
python limpieza.py
```

**Funcionalidades:**
- Calcula tasa de utilidad: `HelpfulnessNumerator / HelpfulnessDenominator`
- Crea etiqueta binaria `IsHelpful` (umbral: 70%)
- Limpia texto: lowercase, URLs, caracteres especiales
- Guarda dataset preparado: `data/amazon_reviews_prepared.csv`

### Paso 3: Extracción de Características NLP

```bash
python nlp_features.py
```

**Características Extraídas:**

**Longitud y Estructura:**
- `char_count`: Número de caracteres
- `word_count`: Número de palabras
- `sentence_count`: Número de oraciones
- `avg_word_length`: Longitud promedio de palabras
- `words_per_sentence`: Palabras por oración

**Léxicas:**
- `exclamation_count`: Exclamaciones
- `question_count`: Preguntas
- `uppercase_word_count`: Palabras en mayúsculas
- `lexical_diversity`: Type-token ratio

**Sentimiento:**
- `vader_neg`, `vader_neu`, `vader_pos`, `vader_compound`: Sentimiento VADER
- `textblob_polarity`: Polaridad (-1 a 1)
- `textblob_subjectivity`: Subjetividad (0 a 1)

**Adicionales:**
- `digit_ratio`: Proporción de dígitos
- `review_score`: Calificación en estrellas

**Salida:** `data/amazon_reviews_with_features.csv`

### Paso 4: Entrenar Modelo

```bash
python model_training.py
```

**Funcionalidades:**
- Entrena modelo LightGBM con características NLP
- Split train/test: 80/20
- Métricas: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Genera gráficos con Plotly:
  - Curva ROC
  - Importancia de características
  - Distribución de probabilidades
- Guarda modelo: `models/review_helpfulness_model_latest.pkl`

**Ejemplo de Salida:**
```
Métricas de evaluación:
  accuracy: 0.8234
  precision: 0.8156
  recall: 0.8312
  f1_score: 0.8233
  roc_auc: 0.8891
```

## 🌐 API REST con FastAPI

### Iniciar API

```bash
python api_app.py
```

La API estará disponible en: `http://localhost:8000`

### Endpoints

#### 1. Health Check

```bash
GET http://localhost:8000/health
```

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/review_helpfulness_model_latest.pkl",
  "features_count": 17
}
```

#### 2. Predicción de Utilidad

```bash
POST http://localhost:8000/reviews/predict_helpfulness
```

**Request Body:**
```json
{
  "text": "This product is amazing! It works exactly as described and the quality is excellent. I highly recommend it to anyone looking for a reliable solution.",
  "score": 5
}
```

**Respuesta:**
```json
{
  "is_helpful_probability": 0.8234,
  "is_helpful": true,
  "confidence": "high",
  "features": {
    "char_count": 156,
    "word_count": 28,
    "sentence_count": 2,
    "vader_compound": 0.8915,
    "textblob_polarity": 0.75,
    ...
  },
  "suggestions": [
    "¡Excelente reseña! Es informativa y probablemente será útil para otros usuarios."
  ]
}
```

#### 3. Información del Modelo

```bash
GET http://localhost:8000/model/info
```

### Documentación Interactiva

FastAPI genera documentación automática:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📱 Dashboard Interactivo

### Iniciar Dashboard

**Importante:** La API debe estar ejecutándose primero.

```bash
# Terminal 1: Iniciar API
python api_app.py

# Terminal 2: Iniciar Dashboard
python dashboard.py
```

Dashboard disponible en: `http://localhost:8050`

### Funcionalidades del Dashboard

1. **Editor de Reseñas:**
   - Selector de calificación (1-5 estrellas)
   - Área de texto para escribir reseña
   - Contador de palabras y caracteres en tiempo real

2. **Análisis en Tiempo Real:**
   - Indicador de utilidad (Útil / Poco Útil)
   - Gráfico gauge con puntuación 0-100%
   - Nivel de confianza de la predicción

3. **Sugerencias Personalizadas:**
   - Recomendaciones para mejorar la reseña
   - Feedback sobre longitud, sentimiento, estructura

4. **Visualización de Características:**
   - Gráfico de barras con características extraídas
   - Valores numéricos de métricas NLP

## 📈 Características del Modelo

### Algoritmo

- **LightGBM** (Gradient Boosting)
  - Rápido y eficiente
  - Maneja bien features numéricas
  - Reduce overfitting con regularización

### Hiperparámetros

```python
{
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8
}
```

### Métricas de Evaluación

- **Accuracy**: Porcentaje de predicciones correctas
- **Precision**: Proporción de predicciones positivas correctas
- **Recall**: Proporción de casos positivos detectados
- **F1-Score**: Media armónica de precision y recall
- **ROC-AUC**: Área bajo la curva ROC

## 🔧 Uso Avanzado

### Entrenar con Dataset Completo

Por defecto, los scripts cargan 50,000 filas para pruebas rápidas. Para entrenar con todo el dataset:

```python
# En limpieza.py, línea 211
df = cargar_datos(DATA_PATH, nrows=None)  # Quitar nrows

# O desde línea de comandos
python limpieza.py --full-dataset
```

### Ajustar Umbral de Utilidad

```python
# En limpieza.py, línea 218
df = calcular_tasa_utilidad(df, umbral=0.6)  # Cambiar umbral
```

### Personalizar Modelo

```python
# En model_training.py
custom_params = {
    'num_leaves': 50,
    'learning_rate': 0.03,
    'max_depth': 10
}
model.entrenar(X_train, y_train, params=custom_params)
```

## 📊 Resultados Esperados

Con el dataset completo (568,454 reseñas), se esperan resultados similares a:

- **ROC-AUC**: ~0.88-0.91
- **Accuracy**: ~0.82-0.85
- **F1-Score**: ~0.81-0.84

### Características Más Importantes

Típicamente, las características más predictivas son:
1. `word_count`: Longitud de la reseña
2. `vader_compound`: Sentimiento general
3. `sentence_count`: Estructura del texto
4. `review_score`: Calificación en estrellas
5. `lexical_diversity`: Variedad de vocabulario

## 🐛 Troubleshooting

### Error: Modelo no encontrado

```bash
# Entrenar el modelo primero
cd scripts
python model_training.py
```

### Error: API no conecta

```bash
# Verificar que la API esté ejecutándose
curl http://localhost:8000/health
```

### Error: Datos no encontrados

```bash
# Descargar dataset de Kaggle
# Colocar Reviews.csv en carpeta data/
```

### Error de NLTK

```python
# Descargar recursos manualmente
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 📚 Recursos

- **Dataset**: [Amazon Fine Food Reviews (Kaggle)](https://www.kaggle.com/snap/amazon-fine-food-reviews)
- **LightGBM**: [Documentación oficial](https://lightgbm.readthedocs.io/)
- **FastAPI**: [Documentación oficial](https://fastapi.tiangolo.com/)
- **Dash**: [Documentación oficial](https://dash.plotly.com/)
- **NLTK**: [Natural Language Toolkit](https://www.nltk.org/)

## 🤝 Contribuciones

Este proyecto es parte de un ejercicio académico de Machine Learning y NLP.

## 📝 Licencia

MIT License - Libre para uso educativo y personal.

## 👥 Autores

Proyecto desarrollado como caso de estudio de Aprendizaje Supervisado y NLP.

---

**¿Preguntas?** Consulta la documentación interactiva de la API en http://localhost:8000/docs
