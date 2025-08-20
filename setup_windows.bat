@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   Configuración de CorrientesAI
echo   Optimizado con Mistral
echo ========================================
echo.

echo 🚀 Iniciando configuración automática...
echo.

REM Verificar si Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python no está instalado
    echo 💡 Descarga Python desde: https://python.org
    pause
    exit /b 1
)

echo ✅ Python detectado

REM Crear entorno virtual si no existe
if not exist ".venv" (
    echo 🔄 Creando entorno virtual...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Error creando entorno virtual
        pause
        exit /b 1
    )
    echo ✅ Entorno virtual creado
) else (
    echo ✅ Entorno virtual ya existe
)

REM Activar entorno virtual
echo 🔄 Activando entorno virtual...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Error activando entorno virtual
    pause
    exit /b 1
)

REM Instalar dependencias
echo 📦 Instalando dependencias...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Error instalando dependencias
    pause
    exit /b 1
)
echo ✅ Dependencias instaladas

REM Configurar Ollama
echo 🤖 Configurando Ollama con Mistral...
python setup_ollama.py
if errorlevel 1 (
    echo ⚠️ Configuración de Ollama incompleta
    echo 💡 Puedes continuar y configurar Ollama manualmente
    echo.
)

echo.
echo ========================================
echo   ¡Configuración completada!
echo ========================================
echo.
echo 🎉 Para iniciar la aplicación:
echo    streamlit run app.py
echo.
echo 💡 Si Ollama no está configurado:
echo    1. Instala Ollama desde: https://ollama.ai
echo    2. Ejecuta: ollama serve
echo    3. Ejecuta: ollama pull mistral
echo.
pause 