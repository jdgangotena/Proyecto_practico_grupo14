"""
Dashboard Interactivo - Review Helpfulness Assistant
Dashboard para escribir reseñas y obtener predicciones de utilidad en tiempo real.
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import requests
import pandas as pd
import os
import sys

# Añadir el directorio scripts al path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "scripts"))

# Configuración
API_URL = "http://localhost:8000"
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

# Inicializar la aplicación Dash con tema Bootstrap
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Review Helpfulness Assistant"
)

# Layout del dashboard
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🔍 Asistente de Reseñas", className="text-center mb-2"),
            html.P(
                "Escribe tu reseña y obtén una puntuación de utilidad en tiempo real",
                className="text-center text-muted mb-4"
            ),
        ])
    ]),

    # Indicador de conexión con la API
    dbc.Row([
        dbc.Col([
            dbc.Alert(
                id="api-status-alert",
                children="Verificando conexión con la API...",
                color="info",
                className="mb-4"
            )
        ])
    ]),

    # Sección principal: Input de reseña
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("✍️ Escribe tu Reseña")),
                dbc.CardBody([
                    # Selector de calificación
                    html.Label("Calificación (estrellas):", className="mb-2"),
                    dcc.Slider(
                        id="rating-slider",
                        min=1,
                        max=5,
                        step=1,
                        value=5,
                        marks={i: f"{i}⭐" for i in range(1, 6)},
                        className="mb-4"
                    ),

                    # Área de texto
                    html.Label("Texto de la reseña:", className="mb-2"),
                    dbc.Textarea(
                        id="review-text",
                        placeholder="Escribe tu reseña aquí... Sé específico y detallado sobre tu experiencia con el producto.",
                        style={"height": "200px"},
                        className="mb-3"
                    ),

                    # Contador de palabras
                    html.Div(id="word-count", className="text-muted mb-3"),

                    # Botón de análisis
                    dbc.Button(
                        "Analizar Reseña",
                        id="analyze-button",
                        color="primary",
                        className="w-100",
                        size="lg"
                    ),
                ])
            ], className="shadow-sm")
        ], width=12, lg=6),

        # Sección de resultados
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("📊 Resultados del Análisis")),
                dbc.CardBody([
                    # Indicador de utilidad
                    html.Div(id="helpfulness-indicator", className="text-center mb-4"),

                    # Gráfico de gauge
                    dcc.Graph(id="helpfulness-gauge", config={'displayModeBar': False}),

                    # Nivel de confianza
                    html.Div(id="confidence-badge", className="text-center mb-3"),

                    # Loading spinner
                    dcc.Loading(
                        id="loading",
                        type="default",
                        children=html.Div(id="loading-output")
                    )
                ])
            ], className="shadow-sm")
        ], width=12, lg=6),
    ], className="mb-4"),

    # Sección de sugerencias
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("💡 Sugerencias para Mejorar")),
                dbc.CardBody([
                    html.Div(id="suggestions-list")
                ])
            ], className="shadow-sm")
        ])
    ], className="mb-4"),

    # Sección de características extraídas
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("🔬 Características Extraídas")),
                dbc.CardBody([
                    dcc.Graph(id="features-chart")
                ])
            ], className="shadow-sm")
        ])
    ], className="mb-4"),

    # Footer con información
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P(
                "Este asistente usa Machine Learning (LightGBM) y NLP para predecir la utilidad de reseñas. "
                "Las predicciones se basan en características como longitud, sentimiento, estructura y vocabulario.",
                className="text-center text-muted small"
            )
        ])
    ])

], fluid=True, style={"padding": "20px"})


# Callbacks
@app.callback(
    Output("api-status-alert", "children"),
    Output("api-status-alert", "color"),
    Input("analyze-button", "n_clicks")
)
def check_api_status(n_clicks):
    """Verifica el estado de la API."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data.get("model_loaded"):
                return "✓ Conectado a la API - Modelo cargado y listo", "success"
            else:
                return "⚠️ API conectada pero el modelo no está disponible. Entrena el modelo primero.", "warning"
    except:
        return "❌ No se puede conectar a la API. Asegúrate de que api_app.py esté ejecutándose en http://localhost:8000", "danger"

    return "Verificando conexión...", "info"


@app.callback(
    Output("word-count", "children"),
    Input("review-text", "value")
)
def update_word_count(text):
    """Actualiza el contador de palabras."""
    if not text:
        return "0 palabras, 0 caracteres"

    word_count = len(text.split())
    char_count = len(text)

    return f"📝 {word_count} palabras, {char_count} caracteres"


@app.callback(
    Output("helpfulness-indicator", "children"),
    Output("helpfulness-gauge", "figure"),
    Output("confidence-badge", "children"),
    Output("suggestions-list", "children"),
    Output("features-chart", "figure"),
    Output("loading-output", "children"),
    Input("analyze-button", "n_clicks"),
    State("review-text", "value"),
    State("rating-slider", "value"),
    prevent_initial_call=True
)
def analyze_review(n_clicks, text, rating):
    """Analiza la reseña y muestra los resultados."""

    # Validaciones
    if not text or len(text.strip()) < 10:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[{
                "text": "Escribe una reseña para ver el análisis",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14}
            }]
        )
        return (
            html.Div("Escribe al menos 10 caracteres", className="text-warning"),
            empty_fig,
            "",
            html.P("Escribe una reseña para obtener sugerencias."),
            empty_fig,
            ""
        )

    # Llamar a la API
    try:
        response = requests.post(
            f"{API_URL}/reviews/predict_helpfulness",
            json={"text": text, "score": rating},
            timeout=10
        )

        if response.status_code != 200:
            error_fig = go.Figure()
            return (
                html.Div("Error al analizar", className="text-danger"),
                error_fig,
                "",
                html.P("Error al obtener sugerencias"),
                error_fig,
                ""
            )

        data = response.json()

        # 1. Indicador de utilidad
        probability = data["is_helpful_probability"]
        is_helpful = data["is_helpful"]

        if is_helpful:
            indicator = html.Div([
                html.H2("✅ Reseña Útil", className="text-success"),
                html.P(f"Probabilidad: {probability*100:.1f}%")
            ])
        else:
            indicator = html.Div([
                html.H2("⚠️ Reseña Poco Útil", className="text-warning"),
                html.P(f"Probabilidad de utilidad: {probability*100:.1f}%")
            ])

        # 2. Gráfico de gauge
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
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))

        # 3. Badge de confianza
        confidence = data["confidence"]
        confidence_colors = {
            "high": "success",
            "medium": "warning",
            "low": "danger"
        }
        confidence_texts = {
            "high": "Alta Confianza",
            "medium": "Confianza Media",
            "low": "Baja Confianza"
        }

        confidence_badge = dbc.Badge(
            f"🎯 {confidence_texts.get(confidence, 'Desconocida')}",
            color=confidence_colors.get(confidence, "secondary"),
            className="p-2",
            style={"fontSize": "1.1rem"}
        )

        # 4. Sugerencias
        suggestions = data["suggestions"]
        suggestions_list = dbc.ListGroup([
            dbc.ListGroupItem([
                html.I(className="bi bi-lightbulb-fill me-2"),
                sugg
            ]) for sugg in suggestions
        ])

        # 5. Gráfico de características
        features = data["features"]

        # Seleccionar características clave para visualizar
        key_features = {
            'word_count': 'Palabras',
            'sentence_count': 'Oraciones',
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

        df_features = pd.DataFrame(feature_data)

        fig_features = px.bar(
            df_features,
            x='Valor',
            y='Característica',
            orientation='h',
            title='Características Principales de tu Reseña',
            color='Valor',
            color_continuous_scale='Viridis'
        )
        fig_features.update_layout(
            showlegend=False,
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        return indicator, fig_gauge, confidence_badge, suggestions_list, fig_features, ""

    except requests.exceptions.ConnectionError:
        error_fig = go.Figure()
        error_fig.update_layout(
            annotations=[{
                "text": "No se puede conectar a la API",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14, "color": "red"}
            }]
        )
        return (
            html.Div("❌ Error de conexión", className="text-danger"),
            error_fig,
            "",
            html.P("Asegúrate de que la API esté ejecutándose (python api_app.py)"),
            error_fig,
            ""
        )
    except Exception as e:
        error_fig = go.Figure()
        return (
            html.Div(f"Error: {str(e)}", className="text-danger"),
            error_fig,
            "",
            html.P(f"Error al procesar: {str(e)}"),
            error_fig,
            ""
        )


# Punto de entrada
if __name__ == "__main__":
    print("="*60)
    print("DASHBOARD - ASISTENTE DE RESEÑAS")
    print("="*60)
    print("\nPor favor, asegúrate de que la API esté ejecutándose:")
    print("  python api_app.py")
    print("\nDashboard disponible en: http://localhost:8050")
    print("\nPresiona Ctrl+C para detener el servidor\n")

    app.run_server(debug=True, host="0.0.0.0", port=8050)
