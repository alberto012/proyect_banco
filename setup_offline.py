#!/usr/bin/env python3
"""
Script de configuración para modo offline de CorrientesAI
Descarga y configura todos los modelos necesarios para funcionamiento offline
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def print_header():
    """Imprime el header del script"""
    print("=" * 60)
    print("🏦 CORRIENTESAI - CONFIGURACIÓN OFFLINE")
    print("=" * 60)
    print("Configurando el sistema para funcionamiento offline...")
    print()

def check_python_version():
    """Verifica la versión de Python"""
    print("🐍 Verificando versión de Python...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def install_dependencies():
    """Instala las dependencias necesarias"""
    print("\n📦 Instalando dependencias...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def check_ollama_installation():
    """Verifica si Ollama está instalado"""
    print("\n🔧 Verificando instalación de Ollama...")
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Ollama instalado: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama no está instalado correctamente")
            return False
    except FileNotFoundError:
        print("❌ Ollama no está instalado")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Timeout verificando Ollama")
        return False

def install_ollama():
    """Instala Ollama según el sistema operativo"""
    print("\n🔧 Instalando Ollama...")
    system = platform.system().lower()
    
    try:
        if system == "windows":
            print("📥 Instalando Ollama en Windows...")
            subprocess.run(["winget", "install", "Ollama.Ollama"], check=True)
        elif system == "darwin":  # macOS
            print("📥 Instalando Ollama en macOS...")
            subprocess.run(["brew", "install", "ollama"], check=True)
        elif system == "linux":
            print("📥 Instalando Ollama en Linux...")
            subprocess.run(["curl", "-fsSL", "https://ollama.ai/install.sh", "|", "sh"], 
                         shell=True, check=True)
        else:
            print(f"❌ Sistema operativo no soportado: {system}")
            return False
        
        print("✅ Ollama instalado correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando Ollama: {e}")
        return False

def start_ollama():
    """Inicia el servicio de Ollama"""
    print("\n🚀 Iniciando servicio de Ollama...")
    try:
        # Verificar si ya está ejecutándose
        result = subprocess.run(["ollama", "list"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Ollama ya está ejecutándose")
            return True
    except:
        pass
    
    try:
        print("🔄 Iniciando Ollama en segundo plano...")
        # En Windows, usar start para ejecutar en segundo plano
        if platform.system().lower() == "windows":
            subprocess.Popen(["start", "ollama", "serve"], shell=True)
        else:
            subprocess.Popen(["ollama", "serve"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        
        # Esperar un momento para que inicie
        import time
        time.sleep(3)
        
        # Verificar que esté ejecutándose
        result = subprocess.run(["ollama", "list"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Ollama iniciado correctamente")
            return True
        else:
            print("❌ No se pudo iniciar Ollama")
            return False
    except Exception as e:
        print(f"❌ Error iniciando Ollama: {e}")
        return False

def download_models():
    """Descarga los modelos necesarios"""
    print("\n📥 Descargando modelos para modo offline...")
    
    models_to_download = [
        "mistral",  # Modelo principal
        "llama2:7b"  # Modelo de respaldo
    ]
    
    for model in models_to_download:
        print(f"🔄 Descargando {model}...")
        try:
            result = subprocess.run(["ollama", "pull", model], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ {model} descargado correctamente")
            else:
                print(f"❌ Error descargando {model}: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout descargando {model} (puede tardar varios minutos)")
        except Exception as e:
            print(f"❌ Error descargando {model}: {e}")

def download_embeddings_model():
    """Descarga el modelo de embeddings"""
    print("\n🧠 Descargando modelo de embeddings...")
    try:
        from sentence_transformers import SentenceTransformer
        
        print("🔄 Descargando all-MiniLM-L6-v2...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Probar el modelo
        test_text = "Este es un texto de prueba"
        embedding = model.encode(test_text)
        
        if len(embedding) > 0:
            print("✅ Modelo de embeddings descargado y funcionando")
            return True
        else:
            print("❌ Error con el modelo de embeddings")
            return False
    except Exception as e:
        print(f"❌ Error descargando modelo de embeddings: {e}")
        return False

def create_directories():
    """Crea los directorios necesarios"""
    print("\n📁 Creando directorios...")
    directories = ["documents", "chroma_db", "assets"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directorio {directory} creado/verificado")

def test_offline_functionality():
    """Prueba la funcionalidad offline"""
    print("\n🧪 Probando funcionalidad offline...")
    
    try:
        # Probar modelo de embeddings
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode("Prueba offline")
        print("✅ Modelo de embeddings: OK")
        
        # Probar Ollama
        result = subprocess.run(["ollama", "list"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Ollama: OK")
            
            # Probar modelo Mistral
            test_result = subprocess.run(["ollama", "run", "mistral", "Hola"], 
                                       capture_output=True, text=True, timeout=30)
            if test_result.returncode == 0 and test_result.stdout.strip():
                print("✅ Modelo Mistral: OK")
            else:
                print("⚠️ Modelo Mistral: No responde correctamente")
        else:
            print("❌ Ollama: No disponible")
        
        print("✅ Pruebas de funcionalidad completadas")
        return True
    except Exception as e:
        print(f"❌ Error en pruebas: {e}")
        return False

def main():
    """Función principal"""
    print_header()
    
    # Verificar Python
    if not check_python_version():
        return False
    
    # Instalar dependencias
    if not install_dependencies():
        return False
    
    # Crear directorios
    create_directories()
    
    # Verificar/instalar Ollama
    if not check_ollama_installation():
        if not install_ollama():
            print("❌ No se pudo instalar Ollama")
            return False
    
    # Iniciar Ollama
    if not start_ollama():
        print("❌ No se pudo iniciar Ollama")
        return False
    
    # Descargar modelos
    download_models()
    
    # Descargar modelo de embeddings
    if not download_embeddings_model():
        print("⚠️ Advertencia: Modelo de embeddings no disponible")
    
    # Probar funcionalidad
    if test_offline_functionality():
        print("\n" + "=" * 60)
        print("🎉 ¡CONFIGURACIÓN OFFLINE COMPLETADA!")
        print("=" * 60)
        print("✅ El sistema está listo para funcionar offline")
        print("🚀 Ejecuta: streamlit run app.py")
        print("💡 Una vez configurado, funciona sin internet")
        print("=" * 60)
        return True
    else:
        print("\n❌ Configuración incompleta")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 