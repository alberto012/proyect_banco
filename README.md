# 🏦 CorrientesAI - Asistente Inteligente Offline

**Tu asistente inteligente para gestión documental que funciona completamente sin conexión a internet.**

## 🚀 Características Principales

### ✅ **Funcionamiento 100% Offline**
- **Chat Inteligente**: Preguntas y respuestas sin internet
- **Búsqueda Semántica**: Encuentra información en documentos
- **Carga de Archivos**: Procesa PDF, DOCX y TXT localmente
- **Sistema de Comentarios**: Gestiona notas y comentarios
- **Base de Datos Local**: Almacenamiento seguro en tu computadora

### 🎯 **Funcionalidades Offline**
- **Modelos Locales**: IA ejecutándose en tu máquina
- **Procesamiento Local**: Sin envío de datos a servidores externos
- **Privacidad Total**: Tus documentos nunca salen de tu computadora
- **Velocidad**: Respuestas instantáneas sin latencia de red

## 📋 Instalación Rápida

### **Opción 1: Configuración Automática (Recomendada)**
```bash
# Clonar el repositorio
git clone <tu-repositorio>
cd mi_agente_rag

# Configuración automática offline
python setup_offline.py
```

### **Opción 2: Configuración Manual**
```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Instalar Ollama
# Windows
winget install Ollama.Ollama

# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# 3. Iniciar Ollama
ollama serve

# 4. Descargar modelos (una sola vez)
ollama pull mistral
ollama pull llama2:7b

# 5. Ejecutar la aplicación
streamlit run app.py
```

## 🔧 Configuración Offline

### **Requisitos del Sistema**
- **RAM**: Mínimo 8GB (recomendado 16GB)
- **Almacenamiento**: 10GB libres para modelos
- **Sistema**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.8 o superior

### **Modelos Descargados**
- **Mistral**: Modelo principal de IA (4.1GB)
- **Llama2**: Modelo de respaldo (3.8GB)
- **all-MiniLM-L6-v2**: Modelo de embeddings (90MB)

## 🎮 Uso

### **1. Interfaz Principal**
- **Estado del Servicio**: Verifica que todo esté funcionando
- **Chat Inteligente**: Haz preguntas sobre tus documentos
- **Subir Archivos**: Agrega documentos al sistema

### **2. Barra Lateral**
- **🔍 Búsqueda**: Busca en documentos directamente
- **📁 Subir archivo**: Agrega nuevos documentos
- **📋 Ver documentos**: Explora documentos cargados
- **💬 Comentarios**: Gestiona notas y comentarios

### **3. Chat Inteligente Offline**
```
Usuario: "¿Qué dice el documento sobre políticas de crédito?"
IA: "Según el documento 'Políticas_2024.pdf', las políticas de crédito establecen que..."
```

## 📁 Formatos Soportados

| Formato | Extensión | Tamaño Máximo | Características |
|---------|-----------|---------------|-----------------|
| PDF | `.pdf` | 50MB | Documentos escaneados y digitales |
| Word | `.docx` | 50MB | Documentos de Microsoft Word |
| Texto | `.txt` | 50MB | Archivos de texto plano |

## 🔍 Búsqueda Semántica

### **Características**
- **Búsqueda por Similitud**: Encuentra contenido relacionado
- **Búsqueda por Palabras Clave**: Búsqueda tradicional
- **Fuentes Documentadas**: Cada respuesta incluye fuentes
- **Resultados Relevantes**: Ordenados por relevancia

### **Ejemplos de Búsqueda**
```
"políticas de crédito" → Encuentra documentos sobre créditos
"fechas importantes" → Encuentra fechas y plazos
"requisitos" → Encuentra requisitos y condiciones
```

## 💬 Sistema de Comentarios

### **Funcionalidades**
- **Agregar Comentarios**: Notas personalizadas
- **Ver Comentarios**: Revisar notas guardadas
- **Búsqueda en Comentarios**: Encuentra notas específicas
- **Limpieza**: Eliminar comentarios antiguos

## 🛠️ Utilidades

### **Gestión del Sistema**
- **Limpiar Chat**: Borrar historial de conversaciones
- **Actualizar Buscador**: Refrescar índices de búsqueda
- **Ver Estado**: Monitorear servicios
- **Configurar Ollama**: Ayuda con configuración

## 🔒 Privacidad y Seguridad

### **Garantías Offline**
- ✅ **Sin Conexión**: Funciona completamente offline
- ✅ **Datos Locales**: Todo se almacena en tu computadora
- ✅ **Sin Tracking**: No hay seguimiento de uso
- ✅ **Sin Análisis**: No se envían datos a terceros

### **Almacenamiento**
- **Documentos**: `./documents/`
- **Base de Datos**: `./chroma_db/`
- **Modelos**: Caché local de Hugging Face
- **Configuración**: Archivos locales

## 🚨 Solución de Problemas

### **Ollama No Responde**
```bash
# Verificar estado
ollama list

# Reiniciar servicio
ollama serve

# Verificar modelos
ollama list
```

### **Modelos No Disponibles**
```bash
# Descargar modelos
ollama pull mistral
ollama pull llama2:7b

# Verificar descarga
ollama list
```

### **Error de Memoria**
- Cerrar otras aplicaciones
- Reiniciar Ollama
- Usar modelo más pequeño: `llama2:7b`

## 📊 Rendimiento

### **Optimizaciones Implementadas**
- **Chunking Inteligente**: División eficiente de documentos
- **Embeddings Optimizados**: Modelo ligero y rápido
- **Búsqueda Vectorial**: Respuestas instantáneas
- **Caché Local**: Modelos precargados

### **Tiempos de Respuesta**
- **Búsqueda**: < 1 segundo
- **Chat**: 2-5 segundos
- **Carga de Archivos**: 5-30 segundos (según tamaño)

## 🔄 Actualizaciones

### **Mantener Actualizado**
```bash
# Actualizar dependencias
pip install -r requirements.txt --upgrade

# Actualizar modelos (opcional)
ollama pull mistral:latest
```

## 📞 Soporte

### **Problemas Comunes**
1. **Ollama no inicia**: Verificar instalación y permisos
2. **Modelos no cargan**: Verificar espacio en disco
3. **Búsqueda lenta**: Verificar RAM disponible
4. **Archivos no cargan**: Verificar formato y tamaño

### **Logs y Diagnóstico**
```bash
# Ver logs de Ollama
ollama logs

# Verificar estado del sistema
python -c "import streamlit; print('Streamlit OK')"
```

## 🎉 ¡Listo para Usar!

Una vez configurado, tu asistente inteligente offline estará listo para:
- 📚 Gestionar documentos de forma inteligente
- 💬 Responder preguntas sobre tu contenido
- 🔍 Buscar información específica
- 💭 Mantener comentarios y notas
- 🔒 Todo funcionando sin internet

**¡Disfruta de tu asistente inteligente completamente offline!** 🚀✨ 