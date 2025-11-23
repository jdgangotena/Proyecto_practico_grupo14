# 📚 Documentación Técnica - Sistema de Predicción de Utilidad de Reseñas

## 📌 Resumen Ejecutivo

Sistema de Machine Learning para predecir la utilidad de reseñas de productos basándose **exclusivamente en características objetivas del texto**. El modelo evalúa la calidad informativa del contenido sin sesgo hacia reseñas positivas o negativas.

---

## 🎯 Filosofía del Sistema

El sistema está diseñado para valorar **reseñas informativas y detalladas**, independientemente de si son positivas o negativas. Una reseña útil es aquella que proporciona información valiosa para otros compradores.

### Principios Fundamentales

1. **Objetividad**: El modelo juzga solo por contenido informativo, no por sentimiento
2. **Independencia de calificación**: Una reseña de 1 estrella puede ser tan útil como una de 5 estrellas
3. **Valoración de detalles**: Reseñas con información específica son consideradas más útiles

---

## 🔬 Características Extraídas (14 Features)

El modelo utiliza **14 características objetivas** divididas en 4 categorías:

### 1. Características de Longitud (5 features)

Miden la extensión y estructura del texto:

| Feature | Descripción | Ejemplo |
|---------|-------------|---------|
| `char_count` | Número total de caracteres | 1250 |
| `word_count` | Número total de palabras | 237 |
| `avg_word_length` | Longitud promedio de palabras | 5.2 |
| `sentence_count` | Número de oraciones | 12 |
| `words_per_sentence` | Palabras por oración (promedio) | 19.75 |

**Justificación**: Reseñas más largas y bien estructuradas tienden a proporcionar más información.

### 2. Características Léxicas (4 features)

Analizan la riqueza del vocabulario y énfasis:

| Feature | Descripción | Ejemplo |
|---------|-------------|---------|
| `exclamation_count` | Número de signos de exclamación | 3 |
| `question_count` | Número de signos de interrogación | 1 |
| `uppercase_word_count` | Palabras en mayúsculas (énfasis) | 2 |
| `lexical_diversity` | Ratio de palabras únicas / total | 0.68 |

**Justificación**: Alta diversidad léxica indica vocabulario rico y descripción detallada.

### 3. Características Básicas (1 feature)

| Feature | Descripción | Ejemplo |
|---------|-------------|---------|
| `digit_ratio` | Proporción de dígitos en el texto | 0.042 |

**Justificación**: Menciones de números indican datos específicos (precios, medidas, tiempos, etc.).

### 4. Características Específicas del Dominio - Amazon Food Reviews (4 features)

Detectan vocabulario y patrones específicos de reseñas de alimentos:

| Feature | Descripción | Ejemplo |
|---------|-------------|---------|
| `specificity_score` | Menciones de sabor, textura y calidad | 5 |
| `has_comparison` | Comparación con otros productos (0/1) | 1 |
| `personal_experience_score` | Pronombres personales + indicadores de tiempo | 8 |
| `price_mention` | Menciona precio/valor (0/1) | 1 |

**Justificación**:
- **Especificidad**: Vocabulario técnico de alimentos (sweet, salty, crunchy, fresh) indica experiencia real
- **Comparaciones**: "Better than Brand X" ayuda a tomar decisiones de compra
- **Experiencia personal**: "I've been using for months" indica uso genuino del producto
- **Precio**: Menciones de valor/precio son útiles para compradores

---

## 🚫 Características Excluidas Intencionalmente

Las siguientes características **NO** se utilizan para evitar sesgos:

### ❌ Características de Sentimiento

- **VADER** (vader_compound, vader_pos, vader_neg, vader_neu)
- **TextBlob** (textblob_polarity, textblob_subjectivity)

**Razón**: Estas características crean sesgo contra reseñas negativas detalladas. Una reseña muy negativa pero informativa es tan útil como una positiva informativa.

### ❌ Calificación del Producto

- **review_score** (1-5 estrellas)

**Razón**: La calificación no indica utilidad del texto. Una reseña de 1 estrella bien justificada es más útil que una de 5 estrellas que solo dice "excelente".

---

## 🧠 Arquitectura del Modelo

### Algoritmo: LightGBM (Gradient Boosting)

**Configuración del modelo:**

```python
params = {
    'objective': 'binary',           # Clasificación binaria
    'metric': 'binary_logloss',      # Métrica de optimización
    'num_leaves': 31,                # Complejidad del árbol
    'learning_rate': 0.05,           # Tasa de aprendizaje
    'feature_fraction': 0.9,         # Fracción de features por árbol
    'bagging_fraction': 0.8,         # Fracción de datos por árbol
    'bagging_freq': 5,               # Frecuencia de bagging
    'verbose': -1                    # Sin output detallado
}
```

### División de Datos

- **Train**: 80% (stratified)
- **Test**: 20% (stratified)
- **Estratificación**: Mantiene proporción de clases útil/no útil

---

## 📊 Definición de Utilidad

### Cálculo de Tasa de Utilidad

```python
HelpfulnessRate = HelpfulnessNumerator / HelpfulnessDenominator
```

- **HelpfulnessNumerator**: Votos de "útil"
- **HelpfulnessDenominator**: Total de votos

### Etiqueta Binaria

```python
IsHelpful = 1 if HelpfulnessRate >= 0.70 else 0
```

- **Útil (1)**: ≥ 70% de votos positivos
- **No útil (0)**: < 70% de votos positivos
- **Filtro**: Solo reseñas con al menos 1 voto

---

## 🔄 Pipeline de Procesamiento

### 1. Carga de Datos (`data_loader.py`)

```bash
python scripts/data_loader.py
```

**Entrada**: `data/Reviews.csv` (dataset de Amazon)

**Salida**: DataFrame validado con estadísticas

**Funciones principales:**
- `cargar_datos()`: Carga CSV
- `validar_columnas()`: Verifica columnas requeridas
- `obtener_estadisticas_basicas()`: Calcula métricas del dataset

### 2. Limpieza (`limpieza.py`)

```bash
python scripts/limpieza.py
```

**Entrada**: Dataset crudo

**Salida**: `data/amazon_reviews_prepared.csv`

**Funciones principales:**
- `limpiar_texto()`: Normaliza texto, elimina HTML
- `calcular_tasa_utilidad()`: Calcula HelpfulnessRate e IsHelpful

### 3. Extracción de Features (`nlp_features.py`)

```bash
python scripts/nlp_features.py
```

**Entrada**: `data/amazon_reviews_prepared.csv`

**Salida**: `data/amazon_reviews_with_features.csv`

**Clase principal**: `NLPFeatureExtractor`

**Métodos:**
- `extraer_longitud_texto()`: 5 features de longitud
- `extraer_caracteristicas_lexicas()`: 4 features léxicas
- `extraer_caracteristicas_adicionales()`: 1 feature adicional
- `extraer_todas_caracteristicas()`: Combina todas las features

### 4. Entrenamiento del Modelo (`model_training.py`)

```bash
python scripts/model_training.py
```

**Entrada**: `data/amazon_reviews_with_features.csv`

**Salida**:
- `models/review_helpfulness_model_latest.pkl`
- `plots/roc_curve.html`
- `plots/feature_importance.html`
- `plots/probability_distribution.html`

**Clase principal**: `ReviewHelpfulnessModel`

**Métodos:**
- `preparar_datos()`: Split train/test
- `entrenar()`: Entrena LightGBM
- `evaluar()`: Calcula métricas de rendimiento
- `guardar()`: Serializa modelo con pickle

---

## 🌐 API REST (FastAPI)

### Iniciar API

```bash
python api_app.py
```

**URL**: http://localhost:8000

**Documentación interactiva**: http://localhost:8000/docs

### Endpoints

#### 1. Health Check

```http
GET /health
```

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "features_count": 10
}
```

#### 2. Predicción de Utilidad

```http
POST /reviews/predict_helpfulness
```

**Request Body:**
```json
{
  "text": "This product is excellent! The quality is outstanding...",
  "score": 5
}
```

**Response:**
```json
{
  "is_helpful": true,
  "is_helpful_probability": 0.856,
  "confidence": "high",
  "features": {
    "char_count": 1250,
    "word_count": 237,
    "sentence_count": 12,
    "lexical_diversity": 0.68,
    ...
  },
  "suggestions": [
    "Excelente nivel de detalle en tu reseña.",
    "Tu reseña tiene una estructura clara con múltiples oraciones."
  ]
}
```

### Niveles de Confianza

| Probabilidad | Nivel | Color |
|--------------|-------|-------|
| ≥ 70% | high | 🟢 Verde |
| 50-70% | medium | 🟡 Amarillo |
| < 50% | low | 🔴 Rojo |

### Sistema de Sugerencias

La API proporciona feedback personalizado:

- **Longitud insuficiente**: "Tu reseña es muy corta. Añade más detalles..."
- **Falta de estructura**: "Considera organizar tu reseña en párrafos..."
- **Falta de datos**: "Incluye información específica como precios, medidas..."
- **Buena calidad**: "Excelente nivel de detalle en tu reseña."

---

## 🖥️ Dashboard Interactivo (Streamlit)

### Iniciar Dashboard

```bash
streamlit run dashboard.py
```

**URL**: http://localhost:8501

### Características del Dashboard

1. **Entrada de Reseña**
   - Área de texto para escribir reseña
   - Selector de calificación (1-5 estrellas)
   - Contador de palabras y caracteres

2. **Resultados de Análisis**
   - Gauge visual de utilidad (0-100%)
   - Nivel de confianza
   - Sugerencias personalizadas
   - Gráfico de características extraídas

3. **Sidebar**
   - Estado de conexión con API
   - Información del modelo
   - Ejemplos pre-cargados

---

## 📈 Métricas de Rendimiento

### Métricas de Clasificación

El modelo es evaluado con las siguientes métricas:

- **Accuracy**: Precisión general
- **Precision**: Proporción de predicciones positivas correctas
- **Recall**: Proporción de casos positivos detectados
- **F1-Score**: Media armónica de precision y recall
- **ROC-AUC**: Área bajo la curva ROC

### Visualizaciones Generadas

1. **ROC Curve** (`plots/roc_curve.html`)
   - Curva ROC con área bajo la curva
   - Punto óptimo de threshold

2. **Feature Importance** (`plots/feature_importance.html`)
   - Importancia relativa de cada característica
   - Ordenadas de mayor a menor impacto

3. **Probability Distribution** (`plots/probability_distribution.html`)
   - Distribución de probabilidades predichas
   - Separación entre clases

---

## 🔧 Configuración Avanzada

### Ajustar Umbral de Utilidad

En `scripts/limpieza.py`:

```python
def calcular_tasa_utilidad(df, umbral=0.7):  # Cambiar 0.7 a otro valor
    ...
```

**Valores sugeridos:**
- `0.6`: Más permisivo (más reseñas marcadas como útiles)
- `0.7`: Estándar (usado actualmente)
- `0.8`: Más estricto (solo reseñas muy valoradas)

### Modificar Hiperparámetros del Modelo

En `scripts/model_training.py`:

```python
default_params = {
    'num_leaves': 31,        # ↑ Mayor complejidad, ↓ Menor complejidad
    'learning_rate': 0.05,   # ↑ Aprendizaje más rápido, ↓ Más lento
    'num_boost_round': 200   # ↑ Más iteraciones, ↓ Menos iteraciones
}
```

### Usar Todo el Dataset

Por defecto se procesan 50,000 filas. Para usar todo el dataset:

```bash
python run_pipeline.py --nrows 0
```

**Advertencia**: Esto puede tomar 30-60 minutos.

---

## 🐛 Troubleshooting

### Problema: "Modelo no encontrado"

**Solución:**
```bash
python run_pipeline.py
```

### Problema: "API no conecta"

**Solución:**
```bash
# Verificar que la API esté corriendo
curl http://localhost:8000/health

# Si no está corriendo, iniciarla
python api_app.py
```

### Problema: "Dataset no encontrado"

**Solución:**
1. Descargar `Reviews.csv` de Kaggle
2. Colocar en carpeta `data/`

### Problema: Error de NLTK

**Solución:**
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

---

## 📦 Dependencias Principales

```
# Machine Learning
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0

# NLP (solo para procesamiento básico)
nltk>=3.8.0

# API
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0

# Dashboard
streamlit>=1.39.0

# Visualización
plotly>=5.14.0
```

**Nota**: VADER y TextBlob están instalados pero no se utilizan en el modelo.

---

## 🎓 Casos de Uso

### Caso 1: Reseña Negativa Detallada

**Entrada:**
```
"I purchased this product for work but it was very disappointing.
The packaging arrived damaged without protective padding. The
materials are low-quality plastic that scratches easily. After 4
days the switch began to fail. The battery lasts only 55 minutes
instead of the advertised 4 hours. Customer support was non-existent."
```

**Resultado esperado**: 70-85% útil

**Razón**: Alto word_count, múltiples sentence_count, datos específicos (digit_ratio), buena estructura.

### Caso 2: Reseña Positiva Corta

**Entrada:**
```
"Excellent product!"
```

**Resultado esperado**: 20-30% útil

**Razón**: Bajo word_count, baja lexical_diversity, sin detalles específicos.

### Caso 3: Reseña Balanceada

**Entrada:**
```
"The product works well for basic tasks. Build quality is decent
but not premium. Battery life is average (about 3-4 hours). Good
value for the price. Some features are missing compared to
competitors."
```

**Resultado esperado**: 60-75% útil

**Razón**: Detalles específicos, estructura clara, información balanceada.

---

## 🔄 Flujo Completo de Uso

### 1. Primera Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt

# Descargar dataset
# (Manual desde Kaggle)

# Ejecutar pipeline completo
python run_pipeline.py
```

### 2. Uso Diario

**Terminal 1 - API:**
```bash
python api_app.py
```

**Terminal 2 - Dashboard:**
```bash
streamlit run dashboard.py
```

**Navegador:**
- Abrir http://localhost:8501
- Escribir reseña
- Obtener predicción en tiempo real

---

## 📊 Interpretación de Resultados

### Probabilidad de Utilidad

| Rango | Interpretación |
|-------|----------------|
| 0-30% | Reseña poco informativa (muy corta o genérica) |
| 30-50% | Reseña moderadamente útil (falta detalle o estructura) |
| 50-70% | Reseña útil (buena longitud y algunos detalles) |
| 70-100% | Reseña muy útil (detallada, estructurada, informativa) |

### Características Importantes

Las features con mayor impacto típicamente son:

1. **word_count**: Reseñas más largas tienden a ser más útiles
2. **lexical_diversity**: Vocabulario rico indica descripción detallada
3. **sentence_count**: Estructura en múltiples oraciones
4. **digit_ratio**: Presencia de datos numéricos específicos

---

## 🚀 Mejoras Futuras Sugeridas

1. **Soporte Multilingüe**: Agregar análisis para español y otros idiomas
2. **Detección de Tópicos**: Identificar temas mencionados (calidad, precio, durabilidad)
3. **Análisis de Comparaciones**: Detectar menciones de productos competidores
4. **Features Semánticas**: Usar embeddings (BERT, GPT) para capturar significado
5. **Feedback Loop**: Aprender de las valoraciones de usuarios del dashboard

---

## 📄 Licencia y Uso

Este sistema está diseñado para uso educativo y de investigación. El dataset Amazon Fine Food Reviews está sujeto a la licencia de Kaggle.

---

## 👥 Soporte

Para preguntas o problemas:
1. Revisar esta documentación
2. Ejecutar `python check_setup.py` para verificar instalación
3. Consultar logs de la API y dashboard para errores específicos

---

**Última actualización**: 2025-11-07
