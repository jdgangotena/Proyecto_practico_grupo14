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

# Configuración de la página
st.set_page_config(
    page_title="Asistente de Reseñas",
    page_icon="🔍",
    layout="wide"
)

# Configuración
##API_URL = "http://m4gk0ko4sccg8gsokco04so0.31.97.10.143.sslip.io"
API_URL = "http://localhost:8000"  # Descomenta para pruebas locales
DATA_PATH = os.path.join("data", "amazon_reviews_with_features.csv")

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

# Función para cargar datos reales
@st.cache_data
def load_data():
    """
    Carga el dataset con características extraídas.
    Usa cache para evitar recargar en cada interacción.

    Returns:
        DataFrame o None si no existe el archivo
    """
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            return df
        except Exception as e:
            st.error(f"Error al cargar datos: {e}")
            return None
    return None

# Header principal
st.title("🔍 Asistente de Análisis de Reseñas")
st.markdown("**Análisis de utilidad de reseñas usando Machine Learning y NLP**")

# Sidebar con estado del sistema
with st.sidebar:
    st.header("⚙️ Estado del Sistema")
    
    api_connected, api_data = check_api()
    
    if api_connected:
        st.success("✓ API Conectada")
        st.success("✓ Modelo Cargado")
        if api_data:
            st.info(f"Características: {api_data.get('features_count', 'N/A')}")
    else:
        st.error("❌ API No Disponible")
        st.warning("Asegúrate de ejecutar: `python api_app.py`")
    
    st.markdown("---")
    st.markdown("### 📚 Información")
    st.markdown("""
    Este asistente usa Machine Learning (LightGBM) 
    y NLP para predecir la utilidad de reseñas.
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Características")
    st.markdown("""
    - **EDA**: Visualización de datos
    - **ML**: Predicción de utilidad
    - **NLP**: Análisis de sentimiento
    """)
    
    st.markdown("---")
    st.markdown("### 📝 Ejemplos Rápidos")
    
    if st.button("Reseña Útil", use_container_width=True):
        st.session_state['example_text'] = "This coffee is excellent! The flavor is rich and smooth, with notes of chocolate and caramel. I've been buying it for months and it always arrives fresh. Much better than other brands I've tried. Great value for the price."
        st.session_state['example_score'] = 5
    
    if st.button("Reseña Corta", use_container_width=True):
        st.session_state['example_text'] = "Good"
        st.session_state['example_score'] = 4

# Crear pestañas principales
tab1, tab2 = st.tabs(["📊 Análisis Exploratorio (EDA)", "🤖 Predicción de Utilidad (ML)"])

# ==================== PESTAÑA 1: EDA ====================
with tab1:
    st.header("📈 Análisis Exploratorio de Datos")
    st.markdown("Visualiza patrones y tendencias en las reseñas analizadas")

    st.markdown("---")

    # Cargar datos reales
    df = load_data()

    if df is None or len(df) == 0:
        st.warning("⚠️ No hay datos disponibles para mostrar")
        st.info("""
        **Para generar datos:**
        1. Ejecuta el pipeline completo: `python run_pipeline.py`
        2. Esto generará el archivo `data/amazon_reviews_with_features.csv`
        3. Recarga esta página para ver los datos reales
        """)
    else:
        # Sección de métricas generales
        st.subheader("📊 Métricas Generales del Sistema")

        col1, col2, col3, col4 = st.columns(4)

        # DATOS REALES: Calcular métricas desde el dataset
        total_reviews = len(df)
        utilidad_rate = df['IsHelpful'].mean() * 100 if 'IsHelpful' in df.columns else 0
        avg_words = df['word_count'].mean() if 'word_count' in df.columns else 0

        with col1:
            st.metric(
                label="Reseñas Analizadas",
                value=f"{total_reviews:,}"
            )

        with col2:
            st.metric(
                label="Tasa de Utilidad",
                value=f"{utilidad_rate:.1f}%"
            )

        with col3:
            # Calcular promedio de sentimiento
            avg_sentiment = df['vader_compound'].mean() if 'vader_compound' in df.columns else 0
            sentiment_label = "Positivo" if avg_sentiment > 0.05 else "Negativo" if avg_sentiment < -0.05 else "Neutral"
            st.metric(
                label="Sentimiento Promedio",
                value=sentiment_label,
                delta=f"{avg_sentiment:.3f}"
            )

        with col4:
            st.metric(
                label="Palabras Promedio",
                value=f"{avg_words:.0f}"
            )

        st.markdown("---")

        # Gráficos con datos reales
        st.subheader("📈 Distribución de Utilidad por Score")

        # DATOS REALES: Agrupar por Score e IsHelpful
        if 'Score' in df.columns and 'IsHelpful' in df.columns:
            df_grouped = df.groupby(['Score', 'IsHelpful']).size().unstack(fill_value=0)
            df_grouped.columns = ['No Útiles', 'Útiles']
            df_grouped = df_grouped.reset_index()

            fig_scores = go.Figure()
            fig_scores.add_trace(go.Bar(
                name='Útiles',
                x=df_grouped['Score'],
                y=df_grouped['Útiles'],
                marker_color='lightgreen'
            ))
            fig_scores.add_trace(go.Bar(
                name='No Útiles',
                x=df_grouped['Score'],
                y=df_grouped['No Útiles'],
                marker_color='lightcoral'
            ))

            fig_scores.update_layout(
                title='Distribución de Reseñas Útiles vs No Útiles por Score',
                xaxis_title='Score (1-5)',
                yaxis_title='Cantidad de Reseñas',
                barmode='group',
                height=400
            )

            st.plotly_chart(fig_scores, use_container_width=True)

        st.markdown("---")

        # Dos columnas para más gráficos
        col_graf1, col_graf2 = st.columns(2)

        with col_graf1:
            st.subheader("💬 Longitud de Reseñas")

            # DATOS REALES: Categorizar por longitud de palabras
            if 'word_count' in df.columns and 'IsHelpful' in df.columns:
                df_length = df.copy()
                df_length['Categoría'] = pd.cut(
                    df_length['word_count'],
                    bins=[0, 20, 50, 100, float('inf')],
                    labels=['Muy Corta (1-20)', 'Corta (21-50)', 'Media (51-100)', 'Larga (101+)']
                )

                df_length_grouped = df_length.groupby('Categoría', observed=True).agg({
                    'word_count': 'count',
                    'IsHelpful': 'mean'
                }).reset_index()
                df_length_grouped.columns = ['Categoría', 'Cantidad', 'Utilidad']
                df_length_grouped['Utilidad'] = (df_length_grouped['Utilidad'] * 100).round(1)

                fig_length = px.bar(
                    df_length_grouped,
                    x='Categoría',
                    y='Cantidad',
                    color='Utilidad',
                    title='Distribución por Longitud de Texto',
                    color_continuous_scale='Blues',
                    text_auto=True
                )
                fig_length.update_layout(showlegend=False)
                st.plotly_chart(fig_length, use_container_width=True)

        with col_graf2:
            st.subheader("😊 Análisis de Sentimiento")

            # DATOS REALES: Categorizar por VADER compound
            if 'vader_compound' in df.columns:
                df_sentiment = df.copy()
                df_sentiment['Sentimiento'] = pd.cut(
                    df_sentiment['vader_compound'],
                    bins=[-1, -0.5, -0.05, 0.05, 0.5, 1],
                    labels=['Muy Negativo', 'Negativo', 'Neutral', 'Positivo', 'Muy Positivo']
                )

                sentiment_counts = df_sentiment['Sentimiento'].value_counts()
                df_sentiment_grouped = pd.DataFrame({
                    'Sentimiento': sentiment_counts.index,
                    'Cantidad': sentiment_counts.values
                })

                fig_sentiment = px.pie(
                    df_sentiment_grouped,
                    values='Cantidad',
                    names='Sentimiento',
                    title='Distribución de Sentimiento en Reseñas',
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)

# ==================== PESTAÑA 2: PREDICCIÓN ML ====================
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
        if not review_text or len(review_text.strip()) < 20:
            with col_output:
                st.warning("⚠️ Escribe al menos 20 caracteres para analizar la reseña")
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
                                "high": "🟢",
                                "medium": "🟡",
                                "low": "🔴"
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
                            'flesch_reading_ease': 'Legibilidad (Flesch)',
                            'gunning_fog': 'Complejidad (Gunning Fog)',
                            'paragraph_count': 'Número de Párrafos',
                            'bullet_point_count': 'Puntos de Lista',
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
