#!/bin/bash

echo ""
echo "========================================"
echo "  Configuración de CorrientesAI"
echo "  Optimizado con Mistral"
echo "========================================"
echo ""

echo "🚀 Iniciando configuración automática..."
echo ""

# Verificar si Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 no está instalado"
    echo "💡 Instala Python desde: https://python.org"
    exit 1
fi

echo "✅ Python detectado: $(python3 --version)"

# Crear entorno virtual si no existe
if [ ! -d ".venv" ]; then
    echo "🔄 Creando entorno virtual..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "❌ Error creando entorno virtual"
        exit 1
    fi
    echo "✅ Entorno virtual creado"
else
    echo "✅ Entorno virtual ya existe"
fi

# Activar entorno virtual
echo "🔄 Activando entorno virtual..."
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "❌ Error activando entorno virtual"
    exit 1
fi

# Instalar dependencias
echo "📦 Instalando dependencias..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Error instalando dependencias"
    exit 1
fi
echo "✅ Dependencias instaladas"

# Configurar Ollama
echo "🤖 Configurando Ollama con Mistral..."
python setup_ollama.py
if [ $? -ne 0 ]; then
    echo "⚠️ Configuración de Ollama incompleta"
    echo "💡 Puedes continuar y configurar Ollama manualmente"
    echo ""
fi

echo ""
echo "========================================"
echo "   ¡Configuración completada!"
echo "========================================"
echo ""
echo "🎉 Para iniciar la aplicación:"
echo "   streamlit run app.py"
echo ""
echo "💡 Si Ollama no está configurado:"
echo "   1. Instala Ollama: curl -fsSL https://ollama.ai/install.sh | sh"
echo "   2. Ejecuta: ollama serve"
echo "   3. Ejecuta: ollama pull mistral"
echo "" 