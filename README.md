# RAG Local - Sistema de Búsqueda Mejorado

## Mejoras Implementadas para Búsqueda Flexible

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

Las mejoras hacen que el sistema sea mucho más flexible y encuentre documentos incluso con variaciones en mayúsculas, acentos, nombres de archivos y términos relacionados. 