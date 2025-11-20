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

## 🗂️ Estructura del Proyecto

```
proyecto/
├── scripts/
│   ├── data_loader.py         # Carga y validación de datos
│   ├── limpieza.py            # Limpieza y cálculo de tasa de utilidad
│   ├── nlp_features.py        # Extracción de características NLP
│   └── model_training.py      # Entrenamiento del modelo LightGBM
├── api_app.py                 # API FastAPI
├── dashboard.py               # Dashboard interactivo con Streamlit
├── requirements.txt           # Dependencias del proyecto
├── Dockerfile                 # Configuración Docker
├── docker-compose.yml         # Orquestación Docker
├── deploy.sh                  # Script de despliegue automático
├── data/                      # Datos (no incluido en repo)
├── models/                    # Modelos entrenados
└── plots/                     # Gráficos generados
```

---

## 🚀 Instalación y Ejecución

### Opción 1: 🐳 Docker (Recomendado para Producción)

#### Pre-requisitos
- Docker instalado y corriendo
- Modelo entrenado en `models/review_helpfulness_model_latest.pkl`

#### Deploy Rápido

```bash
# Opción A: Script automático (más fácil)
./deploy.sh deploy

# Opción B: Docker Compose
docker-compose up -d --build
```

#### Verificar que funciona

```bash
# Health check
curl http://localhost:8000/health

# Documentación interactiva
open http://localhost:8000/docs
```

#### Comandos útiles

```bash
./deploy.sh          # Menú interactivo
./deploy.sh logs     # Ver logs en tiempo real
./deploy.sh restart  # Reiniciar servicio
./deploy.sh stop     # Detener servicio
./deploy.sh health   # Verificar salud

# O con docker-compose:
docker-compose logs -f api
docker-compose restart api
docker-compose down
```

#### Características Docker

- ✅ Imagen multi-stage optimizada (~700 MB)
- ✅ Usuario no-root para seguridad
- ✅ Health checks automáticos
- ✅ Auto-restart en caso de fallos
- ✅ NLTK data precargada
- ✅ Volúmenes para actualizar modelos sin rebuild

#### Actualizar modelo sin rebuild

```bash
# 1. Entrenar nuevo modelo
python scripts/model_training.py

# 2. Reiniciar contenedor (montará el nuevo modelo)
./deploy.sh restart
```

#### Deploy en producción con Nginx

```bash
# Usar configuración de producción
docker-compose -f docker-compose.prod.yml up -d --build
```

Incluye:
- Nginx como reverse proxy
- Configuración SSL/HTTPS
- Rate limiting
- Logs estructurados

---

### Opción 2: 💻 Instalación Local (Desarrollo)

#### 1. Crear entorno virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

#### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

#### 3. Descargar dataset

Descarga el dataset **Amazon Fine Food Reviews** desde Kaggle:
- URL: https://www.kaggle.com/snap/amazon-fine-food-reviews
- Coloca el archivo `Reviews.csv` en la carpeta `data/`

#### 4. Ejecutar pipeline de entrenamiento

```bash
cd scripts

# Paso 1: Cargar y explorar datos
python data_loader.py

# Paso 2: Limpieza y preprocesamiento
python limpieza.py

# Paso 3: Extracción de características NLP
python nlp_features.py

# Paso 4: Entrenar modelo
python model_training.py
```

#### 5. Iniciar API

```bash
# Volver a la raíz del proyecto
cd ..

# Iniciar API
python api_app.py
```

La API estará disponible en: `http://localhost:8000`

#### 6. Iniciar Dashboard (Opcional)

```bash
# En otra terminal, con la API corriendo
python dashboard.py
```

Dashboard disponible en: `http://localhost:8050`

---

## 📊 Pipeline de Datos

### Paso 1: Carga de Datos
**Script:** `scripts/data_loader.py`

- Carga el dataset de reseñas
- Valida columnas requeridas
- Muestra estadísticas básicas
- Calcula tasa de utilidad promedio

### Paso 2: Limpieza y Preprocesamiento
**Script:** `scripts/limpieza.py`

- Calcula tasa de utilidad: `HelpfulnessNumerator / HelpfulnessDenominator`
- Crea etiqueta binaria `IsHelpful` (umbral: 70%)
- Limpia texto: lowercase, URLs, caracteres especiales
- Guarda: `data/amazon_reviews_prepared.csv`

### Paso 3: Extracción de Características NLP
**Script:** `scripts/nlp_features.py`

**Características extraídas:**

| Categoría | Características |
|-----------|----------------|
| **Longitud y Estructura** | `char_count`, `word_count`, `sentence_count`, `avg_word_length`, `words_per_sentence` |
| **Léxicas** | `exclamation_count`, `question_count`, `uppercase_word_count`, `lexical_diversity` |
| **Sentimiento** | `vader_neg`, `vader_neu`, `vader_pos`, `vader_compound`, `textblob_polarity`, `textblob_subjectivity` |
| **Adicionales** | `digit_ratio`, `review_score` |

**Salida:** `data/amazon_reviews_with_features.csv`

### Paso 4: Entrenamiento del Modelo
**Script:** `scripts/model_training.py`

- Algoritmo: **LightGBM** (Gradient Boosting)
- Split: 80/20 (train/test)
- Métricas: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Genera visualizaciones: ROC curve, feature importance, probability distribution
- Guarda modelo: `models/review_helpfulness_model_latest.pkl`

**Ejemplo de salida:**
```
Métricas de evaluación:
  accuracy: 0.8234
  precision: 0.8156
  recall: 0.8312
  f1_score: 0.8233
  roc_auc: 0.8891
```

**Características más importantes:**
1. `word_count`: Longitud de la reseña
2. `vader_compound`: Sentimiento general
3. `sentence_count`: Estructura del texto
4. `review_score`: Calificación en estrellas
5. `lexical_diversity`: Variedad de vocabulario

---

## 🌐 API REST

### Endpoints Disponibles

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
Content-Type: application/json

{
  "text": "This product is amazing! It works exactly as described and the quality is excellent.",
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
    "textblob_polarity": 0.75
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

Devuelve metadatos del modelo cargado, características y métricas de evaluación.

### Documentación Interactiva

FastAPI genera documentación automática:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Ejemplos de uso con cURL

```bash
# Health check
curl http://localhost:8000/health

# Predicción
curl -X POST http://localhost:8000/reviews/predict_helpfulness \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Great product! Highly recommend. Works perfectly and arrived on time.",
    "score": 5
  }'
```

---

## 📱 Dashboard Interactivo

### Funcionalidades

1. **Editor de Reseñas**
   - Selector de calificación (1-5 estrellas)
   - Área de texto para escribir reseña
   - Contador de palabras y caracteres en tiempo real

2. **Análisis en Tiempo Real**
   - Indicador de utilidad (Útil / Poco Útil)
   - Gráfico gauge con puntuación 0-100%
   - Nivel de confianza de la predicción

3. **Sugerencias Personalizadas**
   - Recomendaciones para mejorar la reseña
   - Feedback sobre longitud, sentimiento, estructura

4. **Visualización de Características**
   - Gráfico de barras con características extraídas
   - Valores numéricos de métricas NLP

### Iniciar Dashboard

```bash
# Terminal 1: Iniciar API
python api_app.py

# Terminal 2: Iniciar Dashboard
python dashboard.py
```

Dashboard disponible en: `http://localhost:8050`

---

## ⚙️ Configuración Avanzada

### Variables de Entorno

Crea un archivo `.env` (ver `.env.example`):

```env
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_CORS_ORIGINS=http://localhost:3000,https://tu-dominio.com
LOG_LEVEL=INFO
```

### Entrenar con Dataset Completo

Por defecto, los scripts cargan 50,000 filas. Para usar el dataset completo:

```python
# En limpieza.py
df = cargar_datos(DATA_PATH, nrows=None)  # Quitar nrows
```

### Ajustar Umbral de Utilidad

```python
# En limpieza.py
df = calcular_tasa_utilidad(df, umbral=0.6)  # Cambiar de 0.7 a 0.6
```

### Personalizar Hiperparámetros

```python
# En model_training.py
custom_params = {
    'num_leaves': 50,
    'learning_rate': 0.03,
    'max_depth': 10,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8
}
```

### Usar Gunicorn para Producción

```bash
# Instalar gunicorn
pip install gunicorn

# Ejecutar con múltiples workers
gunicorn api_app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 60
```

---

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

# Ver logs si usa Docker
docker-compose logs api
```

### Error: Puerto 8000 en uso

```bash
# Cambiar puerto en docker-compose.yml
ports:
  - "8001:8000"

# O al ejecutar localmente
uvicorn api_app:app --host 0.0.0.0 --port 8001
```

### Error: Datos no encontrados

Descarga el dataset de Kaggle y colócalo en `data/Reviews.csv`:
https://www.kaggle.com/snap/amazon-fine-food-reviews

### Error de NLTK

```python
# Descargar recursos manualmente
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Docker: Contenedor se reinicia constantemente

```bash
# Ver logs detallados
docker-compose logs --tail=100 api

# Verificar health check
docker inspect review-api | grep -A 10 Health
```

---

## 🌍 Deploy en Producción

### Servicios Cloud Recomendados

#### AWS ECS/Fargate
```bash
docker build -t review-api:latest .
docker tag review-api:latest <account>.dkr.ecr.<region>.amazonaws.com/review-api:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/review-api:latest
```

#### Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/<project-id>/review-api
gcloud run deploy review-api --image gcr.io/<project-id>/review-api --platform managed
```

#### Heroku
```bash
heroku container:push web -a your-app-name
heroku container:release web -a your-app-name
```

#### DigitalOcean App Platform
Conecta tu repositorio GitHub y usa `docker-compose.yml` directamente.

### Mejores Prácticas para Producción

1. **Seguridad**
   - Usuario no-root en contenedor ✅
   - HTTPS/SSL con certificados válidos
   - Variables de entorno para secrets
   - Rate limiting en endpoints

2. **Performance**
   - Usar Gunicorn con múltiples workers
   - Configurar timeouts apropiados
   - Implementar caché para predicciones frecuentes
   - Monitoreo con Prometheus/Grafana

3. **Mantenibilidad**
   - CI/CD con GitHub Actions
   - Versionado de imágenes Docker
   - Logs estructurados (JSON)
   - Backups automáticos de modelos

---

## 📊 Resultados Esperados

Con el dataset completo (568,454 reseñas):

- **ROC-AUC**: ~0.88-0.91
- **Accuracy**: ~0.82-0.85
- **F1-Score**: ~0.81-0.84

---

## 📚 Recursos

- **Dataset**: [Amazon Fine Food Reviews (Kaggle)](https://www.kaggle.com/snap/amazon-fine-food-reviews)
- **LightGBM**: [Documentación oficial](https://lightgbm.readthedocs.io/)
- **FastAPI**: [Documentación oficial](https://fastapi.tiangolo.com/)
- **Streamlit**: [Documentación oficial](https://docs.streamlit.io/)
- **NLTK**: [Natural Language Toolkit](https://www.nltk.org/)
- **Docker**: [Documentación oficial](https://docs.docker.com/)

---

## 🤝 Contribuciones

Este proyecto es parte de un ejercicio académico de Machine Learning y NLP.

## 📝 Licencia

MIT License - Libre para uso educativo y personal.

## 👥 Autores

Proyecto desarrollado como caso de estudio de Aprendizaje Supervisado y NLP.

---

## 🎯 Quick Start

```bash
# 1. Clonar repo y navegar
cd Proyecto_practico_grupo14

# 2. Deploy con Docker (opción más rápida)
./deploy.sh deploy

# 3. Probar API
curl http://localhost:8000/health
open http://localhost:8000/docs

# 4. Hacer predicción
curl -X POST http://localhost:8000/reviews/predict_helpfulness \
  -H "Content-Type: application/json" \
  -d '{"text": "Amazing product!", "score": 5}'
```

**¿Preguntas?** Consulta la documentación interactiva en http://localhost:8000/docs
