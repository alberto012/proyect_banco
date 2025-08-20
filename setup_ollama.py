#!/usr/bin/env python3
"""
Script de configuración automática de Ollama con Mistral
Para CorrientesAI - Asistente Inteligente
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def run_command(command, description, check_output=False):
    """Ejecuta un comando y maneja errores"""
    print(f"🔄 {description}...")
    try:
        if check_output:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            return result.returncode == 0, result.stdout.strip()
        else:
            result = subprocess.run(command, shell=True, timeout=30)
            return result.returncode == 0, ""
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout ejecutando: {command}")
        return False, ""
    except Exception as e:
        print(f"❌ Error ejecutando '{command}': {e}")
        return False, ""

def check_ollama_installed():
    """Verifica si Ollama está instalado"""
    success, output = run_command("ollama --version", "Verificando si Ollama está instalado", check_output=True)
    if success:
        print(f"✅ Ollama está instalado: {output}")
        return True
    else:
        print("❌ Ollama no está instalado")
        return False

def install_ollama():
    """Instala Ollama según el sistema operativo"""
    import platform
    system = platform.system().lower()
    
    print(f"🖥️ Sistema detectado: {system}")
    
    if system == "windows":
        print("📥 Instalando Ollama en Windows...")
        success, _ = run_command("winget install Ollama.Ollama", "Instalando Ollama con winget")
        if not success:
            print("💡 Si winget falla, descarga manualmente desde: https://ollama.ai")
            return False
    elif system == "darwin":  # macOS
        print("📥 Instalando Ollama en macOS...")
        success, _ = run_command("brew install ollama", "Instalando Ollama con Homebrew")
        if not success:
            print("💡 Si Homebrew falla, ejecuta: /bin/bash -c \"$(curl -fsSL https://ollama.ai/install.sh)\"")
            return False
    else:  # Linux
        print("📥 Instalando Ollama en Linux...")
        success, _ = run_command("curl -fsSL https://ollama.ai/install.sh | sh", "Instalando Ollama")
        if not success:
            print("❌ Error instalando Ollama")
            return False
    
    print("✅ Ollama instalado correctamente")
    return True

def start_ollama():
    """Inicia el servicio de Ollama"""
    print("🚀 Iniciando servicio de Ollama...")
    print("💡 Esto abrirá una nueva ventana de terminal. Mantén esa ventana abierta.")
    
    import platform
    system = platform.system().lower()
    
    if system == "windows":
        # En Windows, abrir en nueva ventana
        os.system("start cmd /k ollama serve")
    elif system == "darwin":  # macOS
        os.system("open -a Terminal ollama serve")
    else:  # Linux
        os.system("gnome-terminal -- ollama serve &")
    
    print("⏳ Esperando 5 segundos para que Ollama inicie...")
    time.sleep(5)
    
    # Verificar si Ollama está respondiendo
    for i in range(10):
        success, _ = run_command("ollama list", f"Verificando Ollama (intento {i+1}/10)", check_output=True)
        if success:
            print("✅ Ollama está ejecutándose correctamente")
            return True
        time.sleep(2)
    
    print("⚠️ Ollama no responde. Verifica que esté ejecutándose manualmente.")
    return False

def download_mistral():
    """Descarga el modelo Mistral"""
    print("📥 Descargando modelo Mistral...")
    print("💡 Esto puede tomar varios minutos dependiendo de tu conexión a internet.")
    
    success, _ = run_command("ollama pull mistral", "Descargando Mistral")
    if success:
        print("✅ Mistral descargado correctamente")
        return True
    else:
        print("❌ Error descargando Mistral")
        return False

def verify_setup():
    """Verifica que todo esté configurado correctamente"""
    print("🔍 Verificando configuración...")
    
    # Verificar Ollama
    success, output = run_command("ollama list", "Listando modelos disponibles", check_output=True)
    if not success:
        print("❌ Ollama no está ejecutándose")
        return False
    
    # Verificar si Mistral está disponible
    if "mistral" in output.lower():
        print("✅ Mistral está disponible")
    else:
        print("⚠️ Mistral no está disponible")
        return False
    
    # Probar el modelo
    print("🧪 Probando modelo Mistral...")
    success, response = run_command('ollama run mistral "Hola, ¿estás funcionando correctamente?"', "Probando Mistral", check_output=True)
    if success and response.strip():
        print("✅ Mistral responde correctamente")
        print(f"📝 Respuesta de prueba: {response[:100]}...")
        return True
    else:
        print("❌ Mistral no responde correctamente")
        return False

def main():
    """Función principal del script"""
    print("🚀 Configuración automática de Ollama con Mistral")
    print("=" * 50)
    
    # Paso 1: Verificar si Ollama está instalado
    if not check_ollama_installed():
        print("\n📥 Ollama no está instalado. Instalando...")
        if not install_ollama():
            print("❌ No se pudo instalar Ollama. Instálalo manualmente desde https://ollama.ai")
            return False
        print("✅ Ollama instalado. Reinicia tu terminal y ejecuta este script nuevamente.")
        return False
    
    # Paso 2: Iniciar Ollama
    print("\n🚀 Iniciando Ollama...")
    if not start_ollama():
        print("❌ No se pudo iniciar Ollama. Inícialo manualmente con: ollama serve")
        return False
    
    # Paso 3: Descargar Mistral
    print("\n📥 Descargando Mistral...")
    if not download_mistral():
        print("❌ No se pudo descargar Mistral")
        return False
    
    # Paso 4: Verificar configuración
    print("\n🔍 Verificando configuración...")
    if not verify_setup():
        print("❌ La configuración no está completa")
        return False
    
    print("\n🎉 ¡Configuración completada exitosamente!")
    print("✅ Ollama está ejecutándose")
    print("✅ Mistral está disponible")
    print("✅ El chat inteligente está listo para usar")
    print("\n💡 Ahora puedes ejecutar: streamlit run app.py")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Configuración cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        sys.exit(1) 