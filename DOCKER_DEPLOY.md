# 🐳 Guía de Despliegue con Docker

Esta guía explica cómo construir y desplegar la API de Review Helpfulness usando Docker.

## 📋 Prerrequisitos

1. **Docker instalado** en tu sistema
   - [Descargar Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - Verificar instalación: `docker --version`

2. **Modelo entrenado** en la carpeta `models/`
   - Debe existir: `models/review_helpfulness_model_latest.pkl`
   - Si no existe, ejecuta primero: `python run_pipeline.py`

## 🏗️ Construcción de la Imagen

### Opción 1: Build básico

```bash
docker build -t review-helpfulness-api .
```

### Opción 2: Build con tag específico

```bash
docker build -t review-helpfulness-api:v1.0.0 .
```

### Opción 3: Build sin cache (forzar rebuild completo)

```bash
docker build --no-cache -t review-helpfulness-api .
```

## 🚀 Ejecución del Contenedor

### Modo básico (puerto 8000)

```bash
docker run -p 8000:8000 review-helpfulness-api
```

### Con puerto personalizado

```bash
docker run -p 5000:5000 -e PORT=5000 review-helpfulness-api
```

### Modo detached (background)

```bash
docker run -d -p 8000:8000 --name review-api review-helpfulness-api
```

### Con logs en tiempo real

```bash
docker run -p 8000:8000 review-helpfulness-api
# O si ya está en background:
docker logs -f review-api
```

## 🔍 Verificación del Servicio

### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Respuesta esperada:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "/app/models/review_helpfulness_model_latest.pkl",
  "features_count": 14
}
```

### 2. Probar predicción

```bash
curl -X POST "http://localhost:8000/reviews/predict_helpfulness" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This coffee is amazing! Rich flavor, smooth finish. Best coffee I have had in years. The packaging keeps it fresh and the price is reasonable for the quality.",
    "score": 5
  }'
```

### 3. Ver documentación interactiva

Abre en tu navegador:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🛠️ Comandos Útiles

### Ver contenedores en ejecución

```bash
docker ps
```

### Ver todos los contenedores (incluyendo detenidos)

```bash
docker ps -a
```

### Detener el contenedor

```bash
docker stop review-api
```

### Reiniciar el contenedor

```bash
docker restart review-api
```

### Eliminar el contenedor

```bash
docker rm review-api
```

### Ver logs del contenedor

```bash
docker logs review-api

# Con seguimiento en tiempo real
docker logs -f review-api

# Últimas 100 líneas
docker logs --tail 100 review-api
```

### Acceder a la shell del contenedor

```bash
docker exec -it review-api /bin/bash
```

### Ver uso de recursos

```bash
docker stats review-api
```

## 🌐 Despliegue en Plataformas Cloud

### Railway.app

1. Conecta tu repositorio de GitHub a Railway
2. Railway detectará automáticamente el Dockerfile
3. Variables de entorno necesarias:
   - `PORT` (Railway lo asigna automáticamente)
4. Deploy automático

### Coolify

1. Crea nueva aplicación desde Git
2. Selecciona "Docker" como tipo de build
3. Railway asignará el puerto automáticamente
4. Deploy

### Docker Hub (para compartir la imagen)

```bash
# Login
docker login

# Tag con tu username
docker tag review-helpfulness-api:latest tu-usuario/review-helpfulness-api:latest

# Push
docker push tu-usuario/review-helpfulness-api:latest

# Otros pueden hacer pull
docker pull tu-usuario/review-helpfulness-api:latest
```

## 📊 Estructura de la Imagen

La imagen Docker utiliza **multi-stage build** para optimizar el tamaño:

### Etapa 1: Builder
- Instala dependencias de compilación (gcc, g++, build-essential)
- Instala paquetes de Python en un virtual environment
- Descarga recursos de NLTK
- **Tamaño**: ~1.2 GB

### Etapa 2: Runtime (Final)
- Usa imagen base Python 3.12-slim
- Copia solo el virtual environment y recursos NLTK
- Ejecuta como usuario no-root (`apiuser`)
- **Tamaño final**: ~650 MB

## 🔒 Seguridad

- ✅ Ejecuta como usuario no-root (`apiuser`)
- ✅ Multi-stage build reduce superficie de ataque
- ✅ Dependencias mínimas en imagen final
- ✅ Health checks automáticos
- ✅ Sin archivos innecesarios (gracias a `.dockerignore`)

## 🐛 Troubleshooting

### Error: "Cannot connect to Docker daemon"

**Solución**: Inicia Docker Desktop

```bash
# macOS/Windows: Abrir Docker Desktop
# Linux:
sudo systemctl start docker
```

### Error: "Port already in use"

**Solución**: Cambia el puerto o detén el proceso que lo usa

```bash
# Opción 1: Usar otro puerto
docker run -p 8001:8000 review-helpfulness-api

# Opción 2: Encontrar y matar el proceso
# macOS/Linux:
lsof -ti:8000 | xargs kill -9

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Error: "Model not found"

**Solución**: Asegúrate de que la carpeta `models/` tenga el modelo entrenado

```bash
# Verificar que existe
ls -la models/

# Si no existe, entrenar el modelo primero
python run_pipeline.py
```

### Error al descargar recursos NLTK

Si el healthcheck falla por recursos NLTK faltantes:

```bash
# Acceder al contenedor
docker exec -it review-api /bin/bash

# Descargar manualmente
python -c "import nltk; nltk.download('all')"
```

## 📈 Monitoreo en Producción

### Ver uso de recursos en tiempo real

```bash
docker stats review-api
```

### Configurar límites de recursos

```bash
docker run -d \
  -p 8000:8000 \
  --name review-api \
  --memory="512m" \
  --cpus="1.0" \
  review-helpfulness-api
```

### Auto-restart en caso de falla

```bash
docker run -d \
  -p 8000:8000 \
  --name review-api \
  --restart unless-stopped \
  review-helpfulness-api
```

## 🔄 Actualización de la Imagen

```bash
# 1. Detener el contenedor actual
docker stop review-api

# 2. Eliminar contenedor e imagen antiguos
docker rm review-api
docker rmi review-helpfulness-api

# 3. Rebuild con nueva versión
docker build -t review-helpfulness-api .

# 4. Ejecutar nueva versión
docker run -d -p 8000:8000 --name review-api --restart unless-stopped review-helpfulness-api
```

## 📞 Soporte

Si tienes problemas:
1. Verifica los logs: `docker logs review-api`
2. Verifica el health check: `curl http://localhost:8000/health`
3. Revisa que el modelo exista en `models/`
4. Asegúrate de que Docker Desktop esté corriendo
