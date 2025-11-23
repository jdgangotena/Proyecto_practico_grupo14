"""
Dashboard Streamlit - Review Helpfulness Assistant
Dashboard mejorado con pestañas para EDA y Predicción ML
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

# ============================
# ESTILOS CSS MODERNOS
# ============================
st.markdown(
    """
    <style>
    /* Fondo general y color de texto */
    .stApp {
        background-color: #0a1f44;  /* azul oscuro */
        color: #ffffff;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #142850;
        color: #ffffff;
    }

    /* Header principal */
    .css-18e3th9 h1 {
        color: #f0f0f0;
        font-size: 3rem;
    }

    /* Subtítulos */
    h2, h3, h4, h5 {
        color: #ffffff;
    }

    /* Tabs */
    div[data-baseweb="tab-list"] button {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        background-color: #1b2a5a !important;
        color: #ffffff !important;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0 4px;
    }

    div[data-baseweb="tab-list"] button[data-selected="true"] {
        background-color: #ff6f61 !important; /* color llamativo para la tab activa */
        color: #ffffff !important;
    }

    /* Botones */
    .stButton>button {
        background-color: #ff6f61;
        color: #ffffff;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
    }

    .stButton>button:hover {
        background-color: #ff4b3e;
        color: #ffffff;
    }

    /* Inputs y textareas */
    textarea, input {
        background-color: #1b2a5a !important;
        color: #ffffff !important;
    }

    /* Metricas */
    .stMetric>div>div>div>div>div {
        background-color: #1b2a5a;
        color: #ffffff;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }

    /* Pie charts y gráficos Plotly */
    .plotly-graph-div {
        background-color: #0a1f44 !important;
    }
    
    /* Métricas - texto y valor en blanco */
    .stMetric > div > div > div > div > div > div:first-child,
    .stMetric > div > div > div > div > div > div:last-child {
        color: #ffffff !important;
    }
    
    /* Métricas - texto y valor en blanco */
    .stMetric div[data-testid="stMetricLabel"] {
        color: #ffffff !important;  /* nombre del KPI */
    }

    .stMetric div[data-testid="stMetricValue"] {
        color: #ffffff !important;  /* valor del KPI */
    }
    
    /* Nombre de las métricas en blanco */
    .stMetric > div > div > div > div:first-child {
        color: #ffffff !important;
    }
    
    /* Texto de métricas (nombre y valor) en blanco */
    .stMetric * {
        color: #ffffff !important;
    }
    
    
    /* Placeholder más visible en text_area */
    textarea::placeholder {
        color: #ffffff !important;  /* Gris claro */
        opacity: 1 !important;       /* fuerza que se muestre */
    }
    
    /* Header principal */
    .stApp .css-18e3th9 h1, 
    .stApp h1 {
        color: #ffffff !important;  /* Blanco puro */
    }
    
    /* Barra superior (Toolbar) */
.stAppToolbar {
    background-color: #142850 !important;  /* fondo azul oscuro */
    color: #142850 !important;             /* texto general blanco */
}

/* Botones dentro de la Toolbar */
.stAppToolbar button {
    background-color: #142850 !important; /* fondo blanco */
    color: #142850 !important;            /* texto azul */
    font-weight: bold !important;         /* opcional */
}

/* Iconos dentro de los botones (svg) */
.stAppToolbar button svg {
    color: #ffffff !important; /* iconos blancos */
}

/* Otros spans dentro de la toolbar (sin afectar botones) */
.stAppToolbar > div span {
    color: #ffffff !important;
    }
    
    /* Iconos de ejecución en la Toolbar (nadador, pelotitas, etc.) */
.stAppToolbar svg {
    color: #ffffff !important;  /* iconos blancos */
}

/* Iconos dentro de botones (mantenerlos blancos) */
.stAppToolbar button svg {
    color: #ffffff !important; 
}

    </style>
    """,
    unsafe_allow_html=True
)


# Configuración de la página
st.set_page_config(
    page_title="Asistente de Reseñas",
    page_icon="🔍",
    layout="wide"
)

# Configuración
<<<<<<< HEAD
API_URL = "http://localhost:8000"
DATA_PATH = os.path.join("data", "amazon_reviews_with_features.csv")

# ======================================
# CARGA DE DATA REAL
# ======================================

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

df = load_data()
=======
API_URL = "http://m4gk0ko4sccg8gsokco04so0.31.97.10.143.sslip.io"
>>>>>>> 6f360a86883e747f4d8320a2ec0d8b981e5c5d3e

# Función para verificar la API
def check_api():
    """Verifica la conexión con la API."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return data.get("model_loaded", False), data
        return False, None
    except:
        return False, None

# Header principal
st.title("🔍 Asistente de Análisis de Reseñas")
st.markdown("**Análisis de utilidad de reseñas usando Machine Learning y NLP**")

# Sidebar con estado del sistema
# ======================================
# SIDEBAR
# ======================================

with st.sidebar:
    st.header("⚙️ Estado del Sistema")

    api_connected, api_data = check_api()

    if api_connected:
        st.success("✓ API Conectada")
        st.success("✓ Modelo Cargado")
        st.info(f"Características del modelo: {api_data.get('features_count', 'N/A')}")
    else:
        st.error("❌ API No Disponible")
        st.warning("Ejecuta: `python api_app.py`")

    st.markdown("---")
    st.markdown("### 📂 Estado del Dataset")

    if df is not None:
        st.success(f"✓ Dataset cargado: {len(df):,} filas")
    else:
        st.error("Dataset no encontrado")
        st.warning("Ejecuta el pipeline: `python run_pipeline.py`")

    st.markdown("---")
    st.markdown("### 📝 Ejemplos")
    if st.button("Reseña Útil", use_container_width=True):
        st.session_state['example_text'] = "This coffee is excellent! The flavor is rich and smooth..."
        st.session_state['example_score'] = 5

    if st.button("Reseña Corta", use_container_width=True):
        st.session_state['example_text'] = "Good"
        st.session_state['example_score'] = 4

# Crear pestañas principales
tab1, tab2 = st.tabs([
    "## 📊 Análisis Exploratorio (EDA)", 
    "## 🤖 Predicción de Utilidad (ML)"
    ])

# =====================================================================
# ============================= TAB 1 =================================
# =====================================================================

with tab1:
    st.header("📈 Análisis Exploratorio de Datos")
    st.markdown("Visualiza patrones y tendencias en las reseñas analizadas")

    if df is None:
        st.error("No se puede generar EDA porque no existe el dataset.")
        st.stop()

    # ================================================================
    # MÉTRICAS GENERALES (REEMPLAZADAS CON DATOS REALES)
    # ================================================================

    st.subheader("📊 Métricas Generales del Sistema")

    total_reviews = len(df)
    helpful_rate = df["IsHelpful"].mean() * 100
    avg_words = df["word_count"].mean()
    avg_confidence = "Alta" if helpful_rate > 65 else "Media"

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Reseñas Analizadas", f"{total_reviews:,}")
    col2.metric("Tasa de Utilidad", f"{helpful_rate:.2f}%")
    col3.metric("Confianza Promedio", avg_confidence)
    col4.metric("Palabras Promedio", f"{avg_words:.1f}")

    st.markdown("---")

    # ================================================================
    # DISTRIBUCIÓN DE UTILIDAD POR SCORE
    # ================================================================

    st.subheader("📈 Distribución de Utilidad por Score")

    score_df = df.groupby("Score")["IsHelpful"].agg(["sum", "count"])
    score_df["NoUseful"] = score_df["count"] - score_df["sum"]
    score_df.rename(columns={"sum": "Useful"}, inplace=True)

    fig_scores = go.Figure()
    fig_scores.add_bar(x=score_df.index, y=score_df["Useful"], name="Útiles", marker_color="green")
    fig_scores.add_bar(x=score_df.index, y=score_df["NoUseful"], name="No Útiles", marker_color="red")

    fig_scores.update_layout(
        title="Distribución de Reseñas Útiles vs No Útiles por Score",
        barmode="group",
        xaxis_title="Score (1-5)",
        yaxis_title="Cantidad"
    )

    st.plotly_chart(fig_scores, use_container_width=True)

    st.markdown("---")

    # ================================================================
    # LONGITUD DE RESEÑAS
    # ================================================================

    colA, colB = st.columns(2)

    with colA:
        st.subheader("💬 Longitud de Reseñas")

        df["LengthCategory"] = pd.cut(
            df["word_count"],
            bins=[0, 20, 50, 100, 9999],
            labels=["Muy Corta", "Corta", "Media", "Larga"]
        )

        length_df = df["LengthCategory"].value_counts().reset_index()
        length_df.columns = ["Categoría", "Cantidad"]

        fig_len = px.bar(
            length_df,
            x="Categoría",
            y="Cantidad",
            color="Cantidad",
            title="Distribución por Longitud de Texto",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(fig_len, use_container_width=True)

    # ================================================================
    # ANÁLISIS DE SENTIMIENTO USANDO DATA REAL
    # ================================================================

    with colB:
        st.subheader("😊 Análisis de Sentimiento (VADER)")

        sentiment_df = pd.cut(
            df["vader_compound"],
            bins=[-1, -0.3, 0.3, 1],
            labels=["Negativo", "Neutral", "Positivo"]
        ).value_counts().reset_index()
        sentiment_df.columns = ["Sentimiento", "Cantidad"]

        fig_sent = px.pie(
            sentiment_df,
            names="Sentimiento",
            values="Cantidad",
            title="Distribución de Sentimiento en Reseñas"
        )

        st.plotly_chart(fig_sent, use_container_width=True)

# =====================================================================
# ============================= TAB 2 =================================
# =====================================================================
with tab2:
    st.header("🤖 Predicción de Utilidad de Reseñas")
    st.markdown("Escribe una reseña y obtén una predicción de su utilidad en tiempo real")
    
    # Layout principal con dos columnas
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        st.subheader("✍️ Datos de la Reseña")
        
        # Cargar ejemplos si existen
        default_text = st.session_state.get('example_text', '')
        default_score = st.session_state.get('example_score', 3)
        
        # Limpiar ejemplos después de cargarlos
        if 'example_text' in st.session_state:
            del st.session_state['example_text']
        if 'example_score' in st.session_state:
            del st.session_state['example_score']
        
        # Formulario de entrada
        with st.form("prediction_form"):
            st.write("##### 📝 Texto de la Reseña")
            review_text = st.text_area(
                "Escribe aquí tu reseña",
                value=default_text,
                placeholder="Escribe tu reseña aquí... Sé específico y detallado sobre tu experiencia. Menciona características del producto, calidad, precio, comparaciones, etc.",
                height=200,
                label_visibility="collapsed"
            )
            
            # Contador de palabras
            if review_text:
                word_count = len(review_text.split())
                char_count = len(review_text)
                st.caption(f"📝 {word_count} palabras, {char_count} caracteres")
            else:
                st.caption("📝 0 palabras, 0 caracteres")
            
            st.write("##### ⭐ Calificación del Producto")
            review_score = st.slider(
                "Score (1-5 estrellas)",
                min_value=1,
                max_value=5,
                value=default_score,
                step=1,
                help="Calificación que el usuario le da al producto"
            )
            
            # Mostrar estrellas visuales
            stars = "⭐" * review_score + "☆" * (5 - review_score)
            st.markdown(f"**Calificación seleccionada:** {stars} ({review_score}/5)")
            
            # Botón de submit
            submit_button = st.form_submit_button(
                label="🔍 Analizar Reseña",
                type="primary",
                use_container_width=True
            )
    
    with col_output:
        st.subheader("📊 Resultados del Análisis")
        results_placeholder = st.empty()
    
    # Procesar análisis cuando se presiona el botón
    if submit_button:
        if not review_text or len(review_text.strip()) < 10:
            with col_output:
                st.warning("⚠️ Escribe al menos 10 caracteres para analizar la reseña")
        elif not api_connected:
            with col_output:
                st.error("❌ No se puede conectar a la API. Asegúrate de que esté ejecutándose.")
        else:
            with st.spinner("Analizando reseña..."):
                try:
                    # Preparar datos para enviar
                    payload = {
                        "text": review_text.strip(),
                        "score": int(review_score)
                    }
                    
                    # Debug: mostrar lo que se envía
                    with st.expander("🔍 Debug - Datos enviados a la API"):
                        st.json(payload)
                    
                    # Llamar a la API con el score proporcionado
                    response = requests.post(
                        f"{API_URL}/reviews/predict_helpfulness",
                        json=payload,
                        timeout=10
                    )
                    
                    # Debug: mostrar respuesta
                    with st.expander("🔍 Debug - Respuesta de la API"):
                        st.write(f"Status Code: {response.status_code}")
                        try:
                            st.json(response.json())
                        except:
                            st.text(response.text)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        with results_placeholder.container():
                            # Indicador principal
                            probability = data["is_helpful_probability"]
                            is_helpful = data["is_helpful"]
                            
                            if is_helpful:
                                st.success(f"### ✅ Reseña Útil ({probability*100:.1f}%)")
                            else:
                                st.warning(f"### ⚠️ Reseña Poco Útil ({probability*100:.1f}%)")
                            
                            # Gauge de probabilidad
                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number+delta",
                                value=probability * 100,
                                title={'text': "Puntuación de Utilidad (%)"},
                                delta={'reference': 50},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 30], 'color': "lightgray"},
                                        {'range': [30, 70], 'color': "lightyellow"},
                                        {'range': [70, 100], 'color': "lightgreen"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 50
                                    }
                                }
                            ))
                            fig_gauge.update_layout(height=300)
                            st.plotly_chart(fig_gauge, use_container_width=True)
                            
                            # Nivel de confianza
                            confidence = data["confidence"]
                            confidence_colors = {
                                "Muy alta": "🟩",   # Verde oscuro
                                "alta": "🟢",        # Verde
                                "Medio alta": "🟨", # Amarillo verdoso
                                "Media": "🟡",      # Amarillo
                                "Medio baja": "🟧",  # Naranja
                                "Baja": "🟥",         # Rojo
                                "Muy baja": "🟫"     # Rojo oscuro / marrón oscuro
                            }
                            st.info(f"{confidence_colors.get(confidence, '⚪')} Nivel de Confianza: **{confidence.upper()}**")
                        
                        # Sugerencias de mejora
                        st.markdown("---")
                        st.subheader("💡 Sugerencias para Mejorar tu Reseña")
                        suggestions = data["suggestions"]
                        for i, suggestion in enumerate(suggestions, 1):
                            st.write(f"{i}. {suggestion}")
                        
                        # Características extraídas
                        st.markdown("---")
                        st.subheader("🔬 Características Extraídas del Texto")
                        
                        features = data["features"]
                        key_features = {
                            'word_count': 'Número de Palabras',
                            'sentence_count': 'Número de Oraciones',
                            'vader_compound': 'Sentimiento (VADER)',
                            'textblob_polarity': 'Polaridad (TextBlob)',
                            'lexical_diversity': 'Diversidad Léxica',
                            'exclamation_count': 'Exclamaciones',
                            'question_count': 'Preguntas'
                        }
                        
                        feature_data = []
                        for key, label in key_features.items():
                            if key in features:
                                feature_data.append({
                                    'Característica': label,
                                    'Valor': features[key]
                                })
                        
                        if feature_data:
                            df_features = pd.DataFrame(feature_data)
                            
                            fig_features = px.bar(
                                df_features,
                                x='Valor',
                                y='Característica',
                                orientation='h',
                                title='Características Principales Extraídas',
                                color='Valor',
                                color_continuous_scale='Viridis'
                            )
                            fig_features.update_layout(showlegend=False, height=400)
                            st.plotly_chart(fig_features, use_container_width=True)
                        
                        # Detalles técnicos (expandible)
                        with st.expander("🔍 Ver todas las características técnicas"):
                            st.json(features)
                    
                    else:
                        with col_output:
                            st.error(f"❌ Error {response.status_code}: No se pudo analizar la reseña")
                            try:
                                error_detail = response.json()
                                
                                # Verificar si es un error de NLTK
                                if "nltk_data" in str(error_detail).lower() or "nltk" in str(error_detail).lower():
                                    st.error("### 📚 Faltan recursos de NLTK")
                                    st.warning("""
                                    **Solución:** El servidor necesita descargar recursos de NLTK.
                                    
                                    Ejecuta estos comandos en el servidor donde corre la API:
                                    """)
                                    st.code("""
python -m nltk.downloader vader_lexicon
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
python -m nltk.downloader averaged_perceptron_tagger
                                    """, language="bash")
                                    
                                    st.info("O descarga todos los recursos con:")
                                    st.code("python -m nltk.downloader all", language="bash")
                                else:
                                    st.error(f"**Detalle del error:**")
                                    st.json(error_detail)
                            except:
                                st.error(f"**Respuesta del servidor:**")
                                st.code(response.text)
                
                except requests.exceptions.ConnectionError:
                    with col_output:
                        st.error("❌ No se puede conectar a la API. Ejecuta: `python api_app.py`")
                except requests.exceptions.Timeout:
                    with col_output:
                        st.error("❌ La API tardó demasiado en responder (timeout)")
                except Exception as e:
                    with col_output:
                        st.error(f"❌ Error inesperado: {str(e)}")
                        st.exception(e)

# Footer con tips
st.markdown("---")
st.caption("""
💡 **Tips para escribir reseñas útiles**:
- Menciona características específicas del producto (sabor, textura, calidad, durabilidad)
- Compara con otros productos similares que hayas usado
- Incluye tu experiencia de uso (cuánto tiempo llevas usándolo)
- Menciona si el precio vale la pena en relación a la calidad
- Añade detalles concretos (números, medidas, especificaciones)
- Explica para quién sería ideal este producto
""")
