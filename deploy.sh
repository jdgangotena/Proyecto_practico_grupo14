#!/bin/bash

# ================================================================
# Script de Despliegue - Review Helpfulness Prediction API
# ================================================================

set -e  # Detener en caso de error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funciones auxiliares
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo ""
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
}

# Verificar requisitos
check_requirements() {
    print_header "Verificando requisitos"

    # Verificar Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker no está instalado. Por favor instala Docker primero."
        exit 1
    fi
    print_success "Docker instalado: $(docker --version)"

    # Verificar Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_warning "docker-compose no encontrado, intentando con 'docker compose'"
        DOCKER_COMPOSE="docker compose"
    else
        DOCKER_COMPOSE="docker-compose"
    fi
    print_success "Docker Compose disponible"

    # Verificar que Docker está corriendo
    if ! docker info &> /dev/null; then
        print_error "Docker no está corriendo. Inicia Docker primero."
        exit 1
    fi
    print_success "Docker daemon corriendo"

    # Verificar modelo
    if [ ! -f "models/review_helpfulness_model_latest.pkl" ]; then
        print_warning "Modelo no encontrado en models/review_helpfulness_model_latest.pkl"
        print_info "Asegúrate de entrenar el modelo antes de desplegar"
        read -p "¿Deseas continuar de todos modos? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "Modelo encontrado"
    fi
}

# Build de la imagen
build_image() {
    print_header "Construyendo imagen Docker"
    print_info "Esto puede tomar varios minutos la primera vez..."

    $DOCKER_COMPOSE build --progress=plain

    print_success "Imagen construida exitosamente"
}

# Iniciar servicios
start_services() {
    print_header "Iniciando servicios"

    $DOCKER_COMPOSE up -d

    print_success "Servicios iniciados"
}

# Verificar salud de la API
check_health() {
    print_header "Verificando salud de la API"

    print_info "Esperando a que la API esté lista..."

    MAX_RETRIES=30
    RETRY_COUNT=0

    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            print_success "API está respondiendo correctamente"

            # Mostrar info del health check
            echo ""
            echo "Response del health check:"
            curl -s http://localhost:8000/health | python -m json.tool || echo "No se pudo formatear JSON"
            echo ""
            return 0
        fi

        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo -n "."
        sleep 2
    done

    echo ""
    print_error "La API no respondió después de $MAX_RETRIES intentos"
    print_info "Mostrando logs:"
    $DOCKER_COMPOSE logs --tail=50 api
    return 1
}

# Mostrar información de acceso
show_info() {
    print_header "Información de Acceso"

    echo -e "${GREEN}La API está corriendo exitosamente!${NC}"
    echo ""
    echo "📍 URLs de acceso:"
    echo "   - API Base:        http://localhost:8000"
    echo "   - Health Check:    http://localhost:8000/health"
    echo "   - Documentación:   http://localhost:8000/docs"
    echo "   - ReDoc:           http://localhost:8000/redoc"
    echo ""
    echo "📝 Comandos útiles:"
    echo "   - Ver logs:        $DOCKER_COMPOSE logs -f api"
    echo "   - Detener:         $DOCKER_COMPOSE down"
    echo "   - Reiniciar:       $DOCKER_COMPOSE restart api"
    echo "   - Ver estado:      $DOCKER_COMPOSE ps"
    echo ""
    echo "🧪 Probar la API:"
    echo '   curl -X POST http://localhost:8000/reviews/predict_helpfulness \'
    echo '     -H "Content-Type: application/json" \'
    echo '     -d '"'"'{"text": "Great product! Highly recommended.", "score": 5}'"'"
    echo ""
}

# Limpiar todo
cleanup() {
    print_header "Limpiando recursos Docker"

    print_info "Deteniendo contenedores..."
    $DOCKER_COMPOSE down -v

    read -p "¿Deseas eliminar también las imágenes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Eliminando imágenes..."
        docker rmi proyecto_practico_grupo14_api 2>/dev/null || true
        docker rmi review-api:latest 2>/dev/null || true
    fi

    print_success "Limpieza completada"
}

# Mostrar logs
show_logs() {
    print_header "Mostrando logs de la API"
    $DOCKER_COMPOSE logs -f --tail=100 api
}

# Menú principal
show_menu() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   Review Helpfulness Prediction API - Script Deploy       ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Selecciona una opción:"
    echo ""
    echo "  1) 🚀 Deploy completo (build + start)"
    echo "  2) 🔨 Solo build (construir imagen)"
    echo "  3) ▶️  Solo start (iniciar servicios)"
    echo "  4) 📊 Ver logs"
    echo "  5) 🏥 Check health"
    echo "  6) 🔄 Reiniciar servicios"
    echo "  7) 🛑 Detener servicios"
    echo "  8) 🧹 Limpiar todo"
    echo "  9) ℹ️  Mostrar info"
    echo "  0) ❌ Salir"
    echo ""
    read -p "Opción: " option
    echo ""

    case $option in
        1)
            check_requirements
            build_image
            start_services
            check_health
            show_info
            ;;
        2)
            check_requirements
            build_image
            ;;
        3)
            start_services
            check_health
            show_info
            ;;
        4)
            show_logs
            ;;
        5)
            check_health
            ;;
        6)
            print_info "Reiniciando servicios..."
            $DOCKER_COMPOSE restart api
            check_health
            print_success "Servicios reiniciados"
            ;;
        7)
            print_info "Deteniendo servicios..."
            $DOCKER_COMPOSE down
            print_success "Servicios detenidos"
            ;;
        8)
            cleanup
            ;;
        9)
            show_info
            ;;
        0)
            print_info "Saliendo..."
            exit 0
            ;;
        *)
            print_error "Opción inválida"
            ;;
    esac
}

# Manejo de argumentos de línea de comandos
if [ $# -eq 0 ]; then
    # Sin argumentos, mostrar menú
    while true; do
        show_menu
        echo ""
        read -p "Presiona Enter para volver al menú (o Ctrl+C para salir)..."
    done
else
    # Con argumentos, ejecutar comando directo
    case $1 in
        deploy)
            check_requirements
            build_image
            start_services
            check_health
            show_info
            ;;
        build)
            check_requirements
            build_image
            ;;
        start)
            start_services
            check_health
            show_info
            ;;
        stop)
            print_info "Deteniendo servicios..."
            $DOCKER_COMPOSE down
            print_success "Servicios detenidos"
            ;;
        restart)
            print_info "Reiniciando servicios..."
            $DOCKER_COMPOSE restart api
            check_health
            print_success "Servicios reiniciados"
            ;;
        logs)
            show_logs
            ;;
        health)
            check_health
            ;;
        clean)
            cleanup
            ;;
        info)
            show_info
            ;;
        *)
            echo "Uso: $0 {deploy|build|start|stop|restart|logs|health|clean|info}"
            echo ""
            echo "Comandos disponibles:"
            echo "  deploy  - Build completo + start + health check"
            echo "  build   - Solo construir imagen"
            echo "  start   - Solo iniciar servicios"
            echo "  stop    - Detener servicios"
            echo "  restart - Reiniciar servicios"
            echo "  logs    - Ver logs en tiempo real"
            echo "  health  - Verificar salud de la API"
            echo "  clean   - Limpiar recursos Docker"
            echo "  info    - Mostrar información de acceso"
            echo ""
            echo "Sin argumentos se muestra el menú interactivo"
            exit 1
            ;;
    esac
fi
