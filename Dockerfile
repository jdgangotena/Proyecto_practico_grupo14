# ================================================================
# Dockerfile - Review Helpfulness Prediction API
# ================================================================

# Etapa 1: Builder - Instalar dependencias
FROM python:3.12-slim as builder

# Configurar variables de entorno para Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema necesarias para compilar
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar solo requirements primero para aprovechar cache de Docker
COPY requirements.txt .

# Instalar dependencias de Python en un directorio virtual
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalar dependencias
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Descargar recursos NLTK necesarios
RUN python -c "import nltk; nltk.download('vader_lexicon', download_dir='/opt/nltk_data'); nltk.download('punkt', download_dir='/opt/nltk_data'); nltk.download('stopwords', download_dir='/opt/nltk_data'); nltk.download('wordnet', download_dir='/opt/nltk_data')"


# ================================================================
# Etapa 2: Runtime - Imagen final optimizada
# ================================================================
FROM python:3.12-slim

# Configurar variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    NLTK_DATA="/opt/nltk_data"

# Instalar solo dependencias runtime mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 apiuser && \
    mkdir -p /app /opt/venv /opt/nltk_data && \
    chown -R apiuser:apiuser /app /opt/venv /opt/nltk_data

# Copiar entorno virtual desde builder
COPY --from=builder --chown=apiuser:apiuser /opt/venv /opt/venv
COPY --from=builder --chown=apiuser:apiuser /opt/nltk_data /opt/nltk_data

# Establecer directorio de trabajo
WORKDIR /app

# Copiar código de la aplicación
COPY --chown=apiuser:apiuser api_app.py .
COPY --chown=apiuser:apiuser scripts/ scripts/
COPY --chown=apiuser:apiuser models/ models/

# Cambiar a usuario no-root
USER apiuser

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Comando para ejecutar la aplicación
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000"]
