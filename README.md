# RAG Local - Sistema de Búsqueda Mejorado

Un sistema de búsqueda inteligente basado en RAG (Retrieval-Augmented Generation) que permite buscar y consultar documentos de manera eficiente usando procesamiento de lenguaje natural.

## 📋 Tabla de Contenidos

- [Características](#características)
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Ejecución](#ejecución)
- [Uso del Sistema](#uso-del-sistema)
- [Mejoras Implementadas](#mejoras-implementadas)
- [Solución de Problemas](#solución-de-problemas)
- [Estructura del Proyecto](#estructura-del-proyecto)

## ✨ Características

- 🔍 **Búsqueda semántica avanzada**: Encuentra documentos usando significado, no solo palabras exactas
- 📄 **Soporte múltiples formatos**: PDF, DOCX, TXT y más
- 🧠 **Procesamiento inteligente**: Normalización de texto y variaciones automáticas de palabras
- 💾 **Base de datos local**: ChromaDB para almacenamiento de embeddings
- 🌐 **Interfaz web**: Streamlit para una experiencia de usuario intuitiva
- 🔄 **Búsqueda flexible**: Encuentra documentos con variaciones de palabras y términos relacionados

## 💻 Requisitos del Sistema

### Requisitos Mínimos
- **Sistema Operativo**: Windows 10/11, macOS 10.14+, o Linux (Ubuntu 18.04+)
- **Python**: Versión 3.8 o superior
- **RAM**: Mínimo 4GB (recomendado 8GB+)
- **Espacio en disco**: 2GB libres
- **Procesador**: Dual-core o superior

### Requisitos Recomendados
- **RAM**: 16GB o más
- **Procesador**: Quad-core o superior
- **Espacio en disco**: 5GB libres
- **GPU**: Opcional, para aceleración de procesamiento

## 🚀 Instalación

### Paso 1: Clonar el Repositorio

```bash
# Clonar desde GitHub
git clone https://github.com/tu-usuario/mi_agente_rag.git

# Navegar al directorio del proyecto
cd mi_agente_rag
```

### Paso 2: Configurar Entorno Virtual

#### En Windows:
```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
.venv\Scripts\activate

# Verificar que está activado (debería mostrar (.venv) al inicio)
```

#### En macOS/Linux:
```bash
# Crear entorno virtual
python3 -m venv .venv

# Activar entorno virtual
source .venv/bin/activate

# Verificar que está activado (debería mostrar (.venv) al inicio)
```

### Paso 3: Instalar Dependencias

```bash
# Actualizar pip
pip install --upgrade pip

# Instalar dependencias del proyecto
pip install -r requirements.txt
```

### Paso 4: Verificar Instalación

```bash
# Verificar que Python está disponible
python --version

# Verificar que las dependencias se instalaron correctamente
pip list
```

## ⚙️ Configuración

### Paso 1: Preparar Documentos

1. Coloca tus documentos en la carpeta `documents/`
2. Formatos soportados:
   - PDF (.pdf)
   - Word (.docx, .doc)
   - Texto (.txt)
   - Markdown (.md)

### Paso 2: Configurar Variables de Entorno (Opcional)

Crea un archivo `.env` en la raíz del proyecto:

```env
# Configuraciones opcionales
OPENAI_API_KEY=tu_api_key_si_usas_openai
MODEL_NAME=llama2  # o gpt-3.5-turbo si usas OpenAI
```

## 🏃‍♂️ Ejecución

### Método 1: Ejecución Directa

```bash
# Asegúrate de que el entorno virtual esté activado
# En Windows:
.venv\Scripts\activate

# En macOS/Linux:
source .venv/bin/activate

# Ejecutar la aplicación
streamlit run app.py
```

### Método 2: Ejecución con Parámetros Específicos

```bash
# Ejecutar en puerto específico
streamlit run app.py --server.port 8501

# Ejecutar en modo headless (sin navegador automático)
streamlit run app.py --server.headless true
```

### Paso 3: Acceder a la Aplicación

1. El navegador se abrirá automáticamente
2. Si no se abre, ve a: `http://localhost:8501`
3. La aplicación estará lista para usar

## 🎯 Uso del Sistema

### Interfaz Principal

1. **Sidebar**: Controles y configuración
2. **Área de chat**: Donde escribes tus consultas
3. **Historial**: Conversaciones anteriores
4. **Documentos**: Lista de archivos procesados

### Realizar Consultas

1. **Consulta simple**: Escribe tu pregunta en el chat
2. **Búsqueda específica**: Usa palabras clave relevantes
3. **Consulta compleja**: Formula preguntas detalladas

### Ejemplos de Consultas

```
✅ Consultas efectivas:
- "¿Qué documentos hablan sobre complementaria?"
- "Busca información sobre agencias"
- "¿Cuál es el proceso de evidencia?"
- "Documentos relacionados con demandas"

❌ Evita consultas muy genéricas:
- "Todo"
- "Información"
- "Documentos"
```

### Funciones Especiales

#### 🧹 Limpiar Base de Datos
- **Cuándo usar**: Después de agregar nuevos documentos
- **Qué hace**: Regenera embeddings y limpia cache
- **Ubicación**: Botón en el sidebar

#### 📁 Agregar Documentos
- **Proceso**: Coloca archivos en `documents/`
- **Formato**: PDF, DOCX, TXT
- **Procesamiento**: Automático al cargar la app

## 🔧 Mejoras Implementadas para Búsqueda Flexible

### Problema Original
El sistema no encontraba archivos cuando se buscaban variaciones de palabras. Por ejemplo, buscar "complementaria" no encontraba archivos llamados "COMPLEMENTARIA2" o documentos que contenían "Agencia complementaria".

### Soluciones Implementadas

#### 1. Normalización de Texto
- **Función `normalize_text()`**: Convierte texto a minúsculas, remueve acentos y caracteres especiales
- **Mejora la búsqueda**: Permite encontrar "complementaria" en "COMPLEMENTARIA2"

#### 2. Búsqueda Mejorada (`enhanced_search()`)
- **Búsqueda semántica**: Usa embeddings para búsqueda contextual
- **Búsqueda de texto**: Busca coincidencias exactas en contenido normalizado
- **Variaciones automáticas**: Genera variaciones de palabras clave automáticamente
  - "complementaria" → ["complementaria", "complementarias", "complementario", "complementarios", "complementar", "complementacion", "complementado", "complementada"]
  - "agencia" → ["agencia", "agencias", "agencial"]
  - "evidencia" → ["evidencia", "evidencias", "evidenciar", "evidenciado"]
- **Búsqueda de términos relacionados**: Encuentra "Agencia complementaria" cuando buscas "complementaria"
- **Sistema de scoring**: Prioriza resultados basado en frecuencia y ubicación de coincidencias

#### 3. Mejora en el Chunking
- **Chunks más grandes**: 800 caracteres (antes 500) para mantener contexto
- **Más overlap**: 200 caracteres (antes 100) para no perder palabras clave
- **Separadores inteligentes**: Preserva mejor la estructura del texto

#### 4. Metadata Enriquecida
- **Contenido normalizado**: Se guarda en metadata para búsqueda rápida
- **Palabras clave del archivo**: Extrae palabras clave del nombre del archivo
- **Búsqueda en nombres**: Permite encontrar archivos por su nombre
- **Score de coincidencia**: Calcula relevancia basada en frecuencia y ubicación

#### 5. Retriever Personalizado
- **Clase `EnhancedRetriever`**: Combina búsqueda semántica y de texto
- **Compatibilidad con Pydantic**: Usa `Field` para definir campos correctamente
- **Métodos actualizados**: Usa `_get_relevant_documents` y `_aget_relevant_documents`
- **Mejor cobertura**: Busca en contenido y nombres de archivos
- **Deduplicación**: Evita resultados duplicados

#### 6. Prompt Mejorado
- **Búsqueda exhaustiva**: Instrucciones específicas para buscar variaciones y términos relacionados
- **Reconocimiento de contexto**: Identifica "Agencia complementaria" como relevante para búsquedas de "complementaria"
- **Menor falsos negativos**: Solo responde "no conozco la respuesta" cuando realmente no encuentra nada

### Cómo Usar las Mejoras

1. **Búsqueda normal**: Escribe "complementaria" y encontrará "COMPLEMENTARIA2"
2. **Búsqueda por archivo**: Busca "evidencia agencias" y encontrará el archivo correspondiente
3. **Variaciones automáticas**: El sistema busca automáticamente variaciones de las palabras
4. **Términos relacionados**: Busca "complementaria" y encuentra "Agencia complementaria"

### Limpiar Base de Datos
Si agregas nuevos documentos o quieres aplicar las mejoras a documentos existentes:
1. Haz clic en "🧹 Limpiar base de datos" en el sidebar
2. El sistema regenerará automáticamente los embeddings con las mejoras
3. Se limpiará el cache y se recargará la página

### Archivos Modificados
- `app.py`: Todas las mejoras implementadas aquí
- `chroma_db/`: Base de datos de embeddings (se regenera automáticamente)

### Correcciones Técnicas
- **Error de Pydantic**: Corregido el error "EnhancedRetriever object has no field vectorstore"
- **Compatibilidad**: El retriever ahora funciona correctamente con LangChain y Pydantic
- **Validación**: Campos definidos correctamente usando `Field` de Pydantic
- **Deprecación**: Actualizados métodos a `_get_relevant_documents` y `_aget_relevant_documents`

### Ejemplo de Uso
```
Usuario: "complementaria"
Sistema: Encuentra "EVIDENCIA AGENCIAS COMPLEMENTARIA2.docx" y responde con su contenido
Sistema: También encuentra documentos que contienen "Agencia complementaria"
```

## 🛠️ Solución de Problemas

### Problemas Comunes

#### Error: "ModuleNotFoundError"
```bash
# Solución: Activar entorno virtual
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

#### Error: "Port already in use"
```bash
# Solución: Usar puerto diferente
streamlit run app.py --server.port 8502
```

#### Error: "No documents found"
1. Verifica que hay archivos en `documents/`
2. Haz clic en "🧹 Limpiar base de datos"
3. Reinicia la aplicación

#### Error: "Memory issues"
1. Cierra otras aplicaciones
2. Reduce el tamaño de chunks en `app.py`
3. Usa menos documentos simultáneamente

### Logs y Debugging

```bash
# Ejecutar con logs detallados
streamlit run app.py --logger.level debug

# Ver logs en tiempo real
tail -f ~/.streamlit/logs/streamlit.log
```

## 📁 Estructura del Proyecto

```
mi_agente_rag/
├── app.py                 # Aplicación principal
├── requirements.txt       # Dependencias de Python
├── README.md             # Este archivo
├── .gitignore           # Archivos ignorados por Git
├── .env                 # Variables de entorno (opcional)
├── documents/           # Documentos a procesar
│   ├── *.pdf
│   ├── *.docx
│   └── *.txt
├── assets/              # Recursos estáticos
│   └── logo.png
├── chroma_db/           # Base de datos de embeddings
└── .venv/               # Entorno virtual
```

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📞 Soporte

Si tienes problemas o preguntas:

1. Revisa la sección [Solución de Problemas](#solución-de-problemas)
2. Busca en los [Issues](https://github.com/tu-usuario/mi_agente_rag/issues)
3. Crea un nuevo issue con detalles del problema

---

**Nota**: Las mejoras hacen que el sistema sea mucho más flexible y encuentre documentos incluso con variaciones en mayúsculas, acentos, nombres de archivos y términos relacionados. 