#!/usr/bin/env python3
"""
Script para descargar modelos cuando hay conexión a internet
Ejecuta este script cuando tengas internet para preparar el modo offline
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header():
    """Imprime el header del script"""
    print("=" * 60)
    print("🏦 CORRIENTESAI - DESCARGAR MODELOS")
    print("=" * 60)
    print("Descargando modelos para funcionamiento offline...")
    print("💡 Ejecuta este script cuando tengas conexión a internet")
    print()

def check_internet():
    """Verifica si hay conexión a internet"""
    print("🌐 Verificando conexión a internet...")
    try:
        import urllib.request
        urllib.request.urlopen('http://www.google.com', timeout=3)
        print("✅ Conexión a internet disponible")
        return True
    except:
        print("❌ No hay conexión a internet")
        print("💡 Conecta a internet y vuelve a ejecutar este script")
        return False

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
            
            # Verificar que se guardó en caché local
            cache_path = Path("./embeddings_cache")
            if cache_path.exists():
                print(f"✅ Modelo guardado en caché local: {cache_path}")
            else:
                print("⚠️ Modelo descargado pero no se encontró en caché local")
            
            return True
        else:
            print("❌ Error con el modelo de embeddings")
            return False
    except Exception as e:
        print(f"❌ Error descargando modelo de embeddings: {e}")
        return False

def download_ollama_models():
    """Descarga los modelos de Ollama"""
    print("\n🤖 Descargando modelos de Ollama...")
    
    models_to_download = [
        "mistral",  # Modelo principal
        "llama2:7b"  # Modelo de respaldo
    ]
    
    for model in models_to_download:
        print(f"🔄 Descargando {model}...")
        try:
            result = subprocess.run(["ollama", "pull", model], 
                                  capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                print(f"✅ {model} descargado correctamente")
            else:
                print(f"❌ Error descargando {model}: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout descargando {model} (puede tardar varios minutos)")
        except Exception as e:
            print(f"❌ Error descargando {model}: {e}")

def create_directories():
    """Crea los directorios necesarios"""
    print("\n📁 Creando directorios...")
    directories = ["documents", "chroma_db", "assets", "embeddings_cache"]
    
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
    
    # Verificar internet
    if not check_internet():
        return False
    
    # Crear directorios
    create_directories()
    
    # Descargar modelo de embeddings
    if not download_embeddings_model():
        print("❌ No se pudo descargar el modelo de embeddings")
        return False
    
    # Descargar modelos de Ollama
    download_ollama_models()
    
    # Probar funcionalidad
    if test_offline_functionality():
        print("\n" + "=" * 60)
        print("🎉 ¡DESCARGA DE MODELOS COMPLETADA!")
        print("=" * 60)
        print("✅ Los modelos están listos para funcionamiento offline")
        print("🚀 Ahora puedes desconectar internet y ejecutar:")
        print("   streamlit run app.py")
        print("💡 La aplicación funcionará completamente sin internet")
        print("=" * 60)
        return True
    else:
        print("\n❌ Descarga incompleta")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 