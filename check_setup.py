"""
Script de Verificación - Review Helpfulness Prediction
Verifica que todas las dependencias y configuraciones estén correctas.
"""

import sys
import os

def print_header(text):
    """Imprime un encabezado."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_python_version():
    """Verifica la versión de Python."""
    print("\n[1] Verificando versión de Python...")
    version = sys.version_info
    print(f"    Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("    ❌ ERROR: Se requiere Python 3.8 o superior")
        return False

    print("    ✓ Versión de Python correcta")
    return True

def check_dependencies():
    """Verifica que todas las dependencias estén instaladas."""
    print("\n[2] Verificando dependencias...")

    dependencies = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('lightgbm', 'lightgbm'),
        ('nltk', 'nltk'),
        ('textblob', 'textblob'),
        ('plotly', 'plotly'),
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('dash', 'dash'),
        ('requests', 'requests'),
    ]

    missing = []
    installed = []

    for module, package in dependencies:
        try:
            __import__(module)
            installed.append(package)
            print(f"    ✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"    ❌ {package} - NO INSTALADO")

    if missing:
        print(f"\n    ❌ Faltan {len(missing)} dependencias")
        print("    Instala con: pip install " + " ".join(missing))
        return False

    print(f"\n    ✓ Todas las dependencias instaladas ({len(installed)})")
    return True

def check_nltk_data():
    """Verifica que los datos de NLTK estén descargados."""
    print("\n[3] Verificando datos de NLTK...")

    import nltk

    nltk_data = [
        ('corpora/stopwords', 'stopwords'),
        ('tokenizers/punkt', 'punkt'),
        ('corpora/wordnet', 'wordnet'),
        ('sentiment/vader_lexicon.zip', 'vader_lexicon'),
    ]

    missing = []

    for path, name in nltk_data:
        try:
            nltk.data.find(path)
            print(f"    ✓ {name}")
        except LookupError:
            missing.append(name)
            print(f"    ❌ {name} - NO DESCARGADO")

    if missing:
        print(f"\n    Descargando recursos faltantes...")
        for name in missing:
            try:
                nltk.download(name, quiet=True)
                print(f"    ✓ {name} descargado")
            except:
                print(f"    ❌ Error al descargar {name}")

    print("    ✓ Datos de NLTK verificados")
    return True

def check_directory_structure():
    """Verifica la estructura de directorios."""
    print("\n[4] Verificando estructura de directorios...")

    directories = ['data', 'models', 'plots', 'scripts']
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for directory in directories:
        path = os.path.join(script_dir, directory)
        if os.path.exists(path):
            print(f"    ✓ {directory}/")
        else:
            print(f"    ⚠️  {directory}/ - Creando...")
            os.makedirs(path, exist_ok=True)
            # Crear archivo .gitkeep
            gitkeep = os.path.join(path, '.gitkeep')
            with open(gitkeep, 'w') as f:
                f.write('')

    print("    ✓ Estructura de directorios correcta")
    return True

def check_dataset():
    """Verifica si el dataset existe."""
    print("\n[5] Verificando dataset...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'Reviews.csv')

    if os.path.exists(data_path):
        size_mb = os.path.getsize(data_path) / (1024 * 1024)
        print(f"    ✓ Dataset encontrado ({size_mb:.1f} MB)")
        return True
    else:
        print("    ⚠️  Dataset no encontrado")
        print("    Descarga desde: https://www.kaggle.com/snap/amazon-fine-food-reviews")
        print(f"    Coloca el archivo 'Reviews.csv' en: {os.path.join(script_dir, 'data/')}")
        return False

def check_scripts():
    """Verifica que todos los scripts existan."""
    print("\n[6] Verificando scripts del proyecto...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    scripts = [
        'scripts/data_loader.py',
        'scripts/limpieza.py',
        'scripts/nlp_features.py',
        'scripts/model_training.py',
        'api_app.py',
        'dashboard.py',
        'run_pipeline.py'
    ]

    missing = []

    for script in scripts:
        path = os.path.join(script_dir, script)
        if os.path.exists(path):
            print(f"    ✓ {script}")
        else:
            missing.append(script)
            print(f"    ❌ {script} - NO ENCONTRADO")

    if missing:
        print(f"\n    ❌ Faltan {len(missing)} scripts")
        return False

    print("    ✓ Todos los scripts presentes")
    return True

def main():
    """Función principal."""
    print_header("VERIFICACIÓN DE CONFIGURACIÓN DEL PROYECTO")
    print("Review Helpfulness Prediction System")

    checks = [
        check_python_version(),
        check_dependencies(),
        check_nltk_data(),
        check_directory_structure(),
        check_dataset(),
        check_scripts(),
    ]

    print_header("RESUMEN")

    passed = sum(checks)
    total = len(checks)

    print(f"\nVerificaciones pasadas: {passed}/{total}")

    if passed == total:
        print("\n✓ ¡Todo listo! Puedes ejecutar el proyecto.")
        print("\nPróximos pasos:")
        print("  1. Ejecutar el pipeline completo:")
        print("     python run_pipeline.py")
        print("\n  2. O ejecutar paso a paso:")
        print("     cd scripts")
        print("     python limpieza.py")
        print("     python nlp_features.py")
        print("     python model_training.py")
        print("\n  3. Iniciar API y Dashboard:")
        print("     python api_app.py")
        print("     python dashboard.py")
        return 0
    else:
        print(f"\n⚠️  {total - passed} verificación(es) fallaron.")
        print("Por favor, corrige los errores antes de continuar.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
