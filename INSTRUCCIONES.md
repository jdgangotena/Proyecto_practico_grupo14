# 🚀 Instrucciones de Uso - Dashboard Streamlit

## ⚡ Inicio Rápido

### 1️⃣ Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 2️⃣ Descargar Dataset
1. Descarga desde: https://www.kaggle.com/snap/amazon-fine-food-reviews
2. Coloca `Reviews.csv` en la carpeta `data/`

### 3️⃣ Ejecutar Pipeline Completo
```bash
python run_pipeline.py
```
Esto procesará los datos, extraerá características y entrenará el modelo (~5-10 minutos).

### 4️⃣ Iniciar API (Terminal 1)
```bash
python api_app.py
```
✓ API disponible en: http://localhost:8000

### 5️⃣ Iniciar Dashboard (Terminal 2)
```bash
streamlit run dashboard.py
```
✓ Dashboard disponible en: http://localhost:8501

## 📱 Usar el Dashboard

1. **Abre tu navegador**: http://localhost:8501
2. **Escribe una reseña** en el área de texto
3. **Selecciona calificación** (1-5 estrellas)
4. **Haz clic** en "🔍 Analizar Reseña"
5. **Observa**:
   - Puntuación de utilidad (0-100%)
   - Gauge visual con colores
   - Sugerencias personalizadas
   - Características extraídas

## 💡 Ejemplos de Uso

### Reseña Útil
```
Texto: "This coffee maker is excellent! The brewing temperature is perfect,
       it has a programmable timer, and makes great coffee every morning."
Calificación: 5 ⭐
Resultado esperado: ~85-90% útil
```

### Reseña Corta (Poco Útil)
```
Texto: "Good"
Calificación: 4 ⭐
Resultado esperado: ~20-30% útil
Sugerencia: "Tu reseña es muy corta..."
```

## 🎨 Características del Dashboard Streamlit

✅ **Interfaz moderna y limpia**
✅ **Sidebar con estado de la API**
✅ **Layout de 2 columnas**
✅ **Gráficos interactivos con Plotly**
✅ **Sugerencias en tiempo real**
✅ **Ejemplos pre-cargados**
✅ **Visualización de características**
✅ **Botones de ejemplo rápido**

## 🔧 Verificación

```bash
# Verificar instalación
python check_setup.py

# Verificar API
curl http://localhost:8000/health

# Ver documentación de la API
# http://localhost:8000/docs
```

## 📊 Estructura del Proyecto

```
opiniones_ecommners-1/
├── scripts/
│   ├── data_loader.py       # Carga de datos
│   ├── limpieza.py          # Preprocesamiento
│   ├── nlp_features.py      # Características NLP
│   └── model_training.py    # Entrenamiento LightGBM
├── api_app.py               # API FastAPI
├── dashboard.py             # Dashboard Streamlit ⭐
├── run_pipeline.py          # Pipeline completo
├── examples.py              # Ejemplos de API
└── check_setup.py           # Verificación

data/                        # Datos
models/                      # Modelos entrenados
plots/                       # Gráficos generados
```

## 🐛 Solución de Problemas

### Error: "Streamlit no encontrado"
```bash
pip install streamlit
```

### Error: "API no conecta"
Asegúrate de que la API esté corriendo en otra terminal:
```bash
python api_app.py
```

### Error: "Modelo no encontrado"
Entrena el modelo primero:
```bash
python run_pipeline.py
```

### Error: "Dataset no encontrado"
Descarga `Reviews.csv` y colócalo en `data/`

## 📚 Más Información

- **README completo**: [README.md](README.md)
- **Guía rápida**: [QUICKSTART.md](QUICKSTART.md)
- **Ejemplos de API**: `python examples.py`
- **Documentación API**: http://localhost:8000/docs

## ✅ Checklist

- [ ] Python 3.8+ instalado
- [ ] Dependencias instaladas
- [ ] Dataset en `data/Reviews.csv`
- [ ] Pipeline ejecutado
- [ ] API corriendo (Terminal 1)
- [ ] Dashboard corriendo (Terminal 2)
- [ ] Navegador en http://localhost:8501

---

**¡Listo!** Ahora puedes escribir reseñas y obtener predicciones de utilidad en tiempo real 🎉
