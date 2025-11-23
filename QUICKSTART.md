# 🚀 Guía de Inicio Rápido

Esta guía te ayudará a poner en marcha el sistema de predicción de utilidad de reseñas en 5 minutos.

## 📋 Pre-requisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Dataset de Amazon Reviews (descargable desde Kaggle)

## ⚡ Instalación Rápida

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Verificar instalación

```bash
python check_setup.py
```

Este script verifica:
- ✓ Versión de Python
- ✓ Dependencias instaladas
- ✓ Datos de NLTK
- ✓ Estructura de directorios
- ✓ Scripts del proyecto

### 3. Descargar dataset

1. Ve a: https://www.kaggle.com/snap/amazon-fine-food-reviews
2. Descarga `Reviews.csv`
3. Colócalo en la carpeta `data/`

## 🎯 Ejecución

### Opción A: Pipeline Completo (Recomendado)

Ejecuta todo el flujo con un solo comando:

```bash
python run_pipeline.py
```

Esto ejecutará:
1. Carga de datos
2. Limpieza y preprocesamiento
3. Extracción de características NLP
4. Entrenamiento del modelo

**Tiempo estimado**: 5-10 minutos (con 50,000 filas)

### Opción B: Paso a Paso

Si prefieres ejecutar cada paso manualmente:

```bash
cd scripts

# Paso 1: Cargar datos
python data_loader.py

# Paso 2: Limpiar y preparar
python limpieza.py

# Paso 3: Extraer características
python nlp_features.py

# Paso 4: Entrenar modelo
python model_training.py
```

## 🌐 Usar la API y Dashboard

### 1. Iniciar la API

En una terminal:

```bash
python api_app.py
```

La API estará disponible en: http://localhost:8000

### 2. Iniciar el Dashboard

En **otra terminal** (mientras la API sigue ejecutándose):

```bash
streamlit run dashboard.py
```

El dashboard estará disponible en: http://localhost:8501

### 3. Probar el Sistema

1. Abre tu navegador en http://localhost:8501
2. Escribe una reseña de prueba
3. Selecciona la calificación (1-5 estrellas)
4. Haz clic en "🔍 Analizar Reseña"
5. ¡Observa las predicciones y sugerencias!

## 📊 Ejemplo de Uso

### Ejemplo de Reseña Útil

```
Texto: "This coffee maker is excellent! The brewing temperature is perfect,
it has a programmable timer, and makes great coffee every morning.
The carafe keeps coffee hot for hours. Highly recommend for daily use."

Calificación: 5 estrellas

Resultado esperado: ~85-90% de utilidad
```

### Ejemplo de Reseña Poco Útil

```
Texto: "Good"

Calificación: 5 estrellas

Resultado esperado: ~20-30% de utilidad
Sugerencia: "Tu reseña es muy corta. Añade más detalles..."
```

## 🧪 Probar la API con cURL

```bash
# Health check
curl http://localhost:8000/health

# Predicción
curl -X POST http://localhost:8000/reviews/predict_helpfulness \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Amazing product! Works great and very durable.",
    "score": 5
  }'
```

## 📝 Probar con Python

```python
import requests

# Hacer predicción
response = requests.post(
    "http://localhost:8000/reviews/predict_helpfulness",
    json={
        "text": "This product exceeded my expectations. Great quality!",
        "score": 5
    }
)

result = response.json()
print(f"Probabilidad de utilidad: {result['is_helpful_probability']:.2%}")
print(f"Útil: {result['is_helpful']}")
print(f"Sugerencias: {result['suggestions']}")
```

## 🔧 Opciones Avanzadas

### Entrenar con todo el dataset

Por defecto se procesan 50,000 filas. Para usar todo el dataset:

```bash
python run_pipeline.py --nrows 0
```

**Advertencia**: Esto puede tomar 30-60 minutos dependiendo de tu hardware.

### Omitir entrenamiento

Si solo quieres procesar datos sin entrenar:

```bash
python run_pipeline.py --skip-training
```

## 🐛 Problemas Comunes

### "Modelo no encontrado"

**Solución**: Entrena el modelo primero:
```bash
cd scripts
python model_training.py
```

### "API no conecta"

**Solución**: Asegúrate de que la API esté ejecutándose en otra terminal:
```bash
python api_app.py
```

### "Dataset no encontrado"

**Solución**: Descarga `Reviews.csv` y colócalo en `data/`

### Error de NLTK

**Solución**: Descarga recursos manualmente:
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 📚 Recursos

- **Documentación API**: http://localhost:8000/docs (cuando la API esté ejecutándose)
- **README completo**: [README.md](README.md)
- **Dataset**: https://www.kaggle.com/snap/amazon-fine-food-reviews

## 🎓 Próximos Pasos

Una vez que todo funcione:

1. **Experimenta** con diferentes reseñas en el dashboard
2. **Analiza** los gráficos generados en la carpeta `plots/`
3. **Personaliza** el umbral de utilidad en `limpieza.py`
4. **Ajusta** hiperparámetros del modelo en `model_training.py`
5. **Integra** la API en tus propias aplicaciones

## ✅ Checklist de Verificación

- [ ] Python 3.8+ instalado
- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] Dataset descargado y en `data/Reviews.csv`
- [ ] Pipeline ejecutado (`python run_pipeline.py`)
- [ ] Modelo entrenado (archivo en `models/`)
- [ ] API funcionando (`python api_app.py`)
- [ ] Dashboard funcionando (`streamlit run dashboard.py`)

---

**¿Necesitas ayuda?** Revisa el [README.md](README.md) completo o ejecuta `python check_setup.py`
