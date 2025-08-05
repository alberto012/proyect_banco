# RAG Local - Agente de Documentos Inteligente

Un agente RAG (Retrieval-Augmented Generation) local que permite consultar documentos PDF y Word de forma offline, con capacidades avanzadas de análisis y comparación de documentos.

## 🚀 Características Principales

### 📁 Subida Inteligente de Archivos
- **Análisis automático**: Compara automáticamente archivos nuevos con documentos existentes
- **Detección de duplicados**: Identifica archivos idénticos para evitar redundancias
- **Análisis semántico**: Detecta cambios significativos usando análisis de texto avanzado
- **Comentarios automáticos**: Genera comentarios detallados sobre los cambios detectados
- **Validaciones robustas**: Verifica formato, tamaño y contenido de archivos

### 🔍 Búsqueda Mejorada
- **Búsqueda semántica**: Utiliza embeddings para encontrar información relevante
- **Búsqueda por texto**: Combina búsqueda semántica con búsqueda de texto normalizado
- **Variaciones de consulta**: Busca automáticamente variaciones y términos relacionados
- **Fuentes detalladas**: Muestra exactamente de dónde proviene cada respuesta

### 💬 Sistema de Comentarios
- **Comentarios automáticos**: Generados automáticamente al subir archivos
- **Comentarios personalizados**: Permite agregar comentarios manuales
- **Historial de comentarios**: Visualiza todos los comentarios guardados
- **Edición de comentarios**: Modifica comentarios automáticos antes de guardarlos

### 📊 Análisis de Documentos
- **Comparación de contenido**: Detecta adiciones, eliminaciones y cambios
- **Análisis de similitud**: Calcula porcentajes de similitud entre documentos
- **Palabras clave**: Identifica términos nuevos y removidos
- **Recomendaciones**: Sugiere acciones basadas en el análisis

## 🛠️ Instalación

### Requisitos Previos
- Python 3.8 o superior
- Ollama instalado y ejecutándose localmente
- Modelo de lenguaje compatible con Ollama (recomendado: llama2, mistral, o codellama)

### Pasos de Instalación

1. **Clonar el repositorio**
   ```bash
   git clone <url-del-repositorio>
   cd mi_agente_rag
   ```

2. **Crear entorno virtual**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar Ollama**
   ```bash
   # Instalar un modelo (ejemplo con llama2)
   ollama pull llama2
   ```

5. **Ejecutar la aplicación**
   ```bash
   streamlit run app.py
   ```

## 📖 Uso

### Subir Archivos

1. **Acceder a la función de subida**
   - Haz clic en "📁 Subir archivo" en la barra lateral

2. **Seleccionar archivo**
   - Arrastra o selecciona un archivo PDF o DOCX
   - El sistema validará automáticamente el formato y tamaño

3. **Análisis automático**
   - El sistema analizará el contenido del archivo
   - Comparará con documentos existentes
   - Generará un comentario automático con los hallazgos

4. **Revisar resultados**
   - **Documentos similares**: Ver documentos existentes relacionados
   - **Análisis de cambios**: Detalles sobre modificaciones detectadas
   - **Comentario automático**: Análisis generado automáticamente

5. **Tomar acciones**
   - **Guardar archivo**: Agregar el documento a la base de datos
   - **Guardar comentario**: Guardar el comentario automático
   - **Editar comentario**: Modificar el comentario antes de guardarlo
   - **Agregar comentario personalizado**: Crear un comentario manual

### Consultar Documentos

1. **Hacer preguntas**
   - Escribe preguntas en lenguaje natural
   - El sistema buscará en todos los documentos y comentarios

2. **Revisar fuentes**
   - Cada respuesta incluye las fuentes utilizadas
   - Puedes ver el contenido exacto de donde proviene la información

### Gestionar Comentarios

1. **Ver comentarios**
   - Haz clic en "📋 Ver comentarios" en la barra lateral
   - Revisa todos los comentarios guardados

2. **Agregar comentarios**
   - Haz clic en "📝 Añadir comentario"
   - Escribe comentarios personalizados

## 🔧 Configuración Avanzada

### Variables de Entorno
```bash
# Configurar modelo de Ollama (opcional)
export OLLAMA_MODEL=llama2

# Configurar directorios (opcional)
export DOCUMENTS_DIR=./documents
export CHROMA_DB_DIR=./chroma_db
```

### Personalización del Modelo
Puedes cambiar el modelo de Ollama modificando la función `get_ollama_llm()` en `app.py`:

```python
def get_ollama_llm():
    return Ollama(model="tu-modelo-aqui")
```

## 📁 Estructura del Proyecto

```
mi_agente_rag/
├── app.py                 # Aplicación principal
├── requirements.txt       # Dependencias
├── README.md             # Documentación
├── .gitignore           # Archivos ignorados por Git
├── documents/           # Directorio de documentos
├── chroma_db/          # Base de datos vectorial
├── assets/             # Recursos estáticos
└── .venv/              # Entorno virtual
```

## 🎯 Características Técnicas

### Análisis de Documentos
- **Normalización de texto**: Mejora la precisión de búsquedas
- **Detección de similitud**: Usa algoritmos de comparación avanzados
- **Análisis semántico**: Identifica cambios significativos en el contenido
- **Extracción de palabras clave**: Detecta términos importantes

### Búsqueda Inteligente
- **Embeddings**: Usa SentenceTransformers para búsqueda semántica
- **Búsqueda híbrida**: Combina embeddings con búsqueda de texto
- **Variaciones automáticas**: Busca términos relacionados automáticamente
- **Ranking inteligente**: Ordena resultados por relevancia

### Manejo de Errores
- **Validaciones robustas**: Verifica archivos antes del procesamiento
- **Manejo de excepciones**: Captura y maneja errores de forma elegante
- **Limpieza automática**: Elimina archivos temporales y parciales
- **Feedback detallado**: Proporciona información clara sobre errores

## 🚨 Solución de Problemas

### Problemas Comunes

1. **Ollama no responde**
   - Verifica que Ollama esté ejecutándose: `ollama serve`
   - Confirma que el modelo esté instalado: `ollama list`

2. **Error al subir archivos**
   - Verifica el formato (solo PDF y DOCX)
   - Confirma que el archivo no esté corrupto
   - Verifica el espacio en disco

3. **Búsquedas lentas**
   - La primera búsqueda puede ser lenta (carga de embeddings)
   - Las búsquedas posteriores son más rápidas

4. **Archivos no encontrados**
   - Verifica que los archivos estén en el directorio `documents/`
   - Confirma que los archivos sean legibles

### Logs y Debugging
Los errores se muestran en la consola donde ejecutas Streamlit. Para más detalles, revisa:
- Mensajes de error en la interfaz
- Logs en la consola
- Archivos temporales en el directorio `documents/`

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🆘 Soporte

Si tienes problemas o preguntas:
1. Revisa la documentación
2. Busca en los issues existentes
3. Crea un nuevo issue con detalles del problema

---

**Desarrollado con ❤️ para análisis inteligente de documentos** 