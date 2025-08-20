# =============================================================================
# IMPORTS - Organizados por categorías
# =============================================================================

# Streamlit y UI
import streamlit as st
from datetime import datetime
from pathlib import Path
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import nest_asyncio
import re
from html import unescape
import unicodedata
import hashlib
import difflib
import shutil

# LangChain Core
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from typing import List, Any
from pydantic import Field

# LangChain Community
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================

# Directorios principales
DOCUMENTS_DIR = Path("./documents")
CHROMA_DB_DIR = Path("./chroma_db")

# Configuración de la aplicación
APP_CONFIG = {
    "page_title": "CorrientesAI - Asistente Inteligente Offline",
    "page_icon": "🏦",
    "max_file_size": 50 * 1024 * 1024,  # 50MB
    "supported_formats": ['.pdf', '.docx', '.txt'],  # Agregado .txt para más compatibilidad
    "embedding_model": "all-MiniLM-L6-v2",  # Modelo local que se descarga una vez
    "llm_model": "mistral",  # Modelo local de Ollama
    "retriever_k": 4,  # Aumentado para mejor cobertura
    "brand_name": "CorrientesAI Offline",
    "brand_description": "Tu asistente inteligente offline para la gestión documental",
    "primary_color": "#1e3a8a",  # Azul corporativo
    "secondary_color": "#3b82f6",  # Azul más claro
    "accent_color": "#f59e0b",  # Dorado para acentos
    "success_color": "#10b981",  # Verde para éxito
    "warning_color": "#f59e0b",  # Amarillo para advertencias
    "error_color": "#ef4444",  # Rojo para errores
    "offline_mode": True  # Modo offline activado
}

# =============================================================================
# EXCEPCIONES PERSONALIZADAS
# =============================================================================

class DiskSpaceError(Exception):
    """Excepción para errores de espacio en disco"""
    pass

class DocumentProcessingError(Exception):
    """Excepción para errores de procesamiento de documentos"""
    pass

# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================

def setup_page_config():
    """Configura la página de Streamlit con diseño limpio"""
    st.set_page_config(
        page_title=APP_CONFIG["page_title"],
        page_icon=APP_CONFIG["page_icon"],
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Cargar archivos del loader personalizado
    def load_css():
        with open('assets/loader.css', 'r', encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    def load_js():
        with open('assets/loader.js', 'r', encoding='utf-8') as f:
            st.markdown(f'<script>{f.read()}</script>', unsafe_allow_html=True)
    
    # Cargar CSS y JS del loader
    try:
        load_css()
        load_js()
    except FileNotFoundError:
        # Si no encuentra los archivos, usar loader básico
        st.markdown("""
        <style>
        .loader-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        }
        .loader-text {
            color: white;
            font-size: 24px;
            font-weight: 600;
            text-align: center;
        }
        .loader-spinner {
            width: 40px;
            height: 40px;
            border: 4px solid rgba(255,255,255,0.3);
            border-top: 4px solid white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 20px 0;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """, unsafe_allow_html=True)
    
    # CSS personalizado para diseño moderno y etéreo
    st.markdown("""
    <style>
    /* Configuración general */
    .main {
        padding-top: 1rem;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        min-height: 100vh;
    }
    
    /* Header principal */
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem;
        background: linear-gradient(135deg, """ + APP_CONFIG["primary_color"] + """ 0%, """ + APP_CONFIG["secondary_color"] + """ 100%);
        border-radius: 20px;
        color: white;
        box-shadow: 0 20px 40px rgba(30, 58, 138, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
        opacity: 0.3;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 800;
        position: relative;
        z-index: 1;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.95;
        position: relative;
        z-index: 1;
        font-weight: 300;
    }
    
    /* Contenedores principales */
    .content-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .content-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.12);
    }
    
    /* Botones modernos */
    .stButton > button {
        border-radius: 12px;
        border: 2px solid """ + APP_CONFIG["secondary_color"] + """;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, """ + APP_CONFIG["secondary_color"] + """, """ + APP_CONFIG["primary_color"] + """);
        color: white;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
        border-color: """ + APP_CONFIG["primary_color"] + """;
    }
    
    /* Métricas modernas */
    .metric-card {
        background: linear-gradient(135deg, """ + APP_CONFIG["primary_color"] + """ 0%, """ + APP_CONFIG["secondary_color"] + """ 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(30, 58, 138, 0.15);
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        transform: rotate(45deg);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover::before {
        transform: rotate(45deg) scale(1.2);
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(30, 58, 138, 0.25);
    }
    
    .metric-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }
    
    .metric-card p {
        margin: 0;
        font-size: 1.4rem;
        font-weight: 700;
        position: relative;
        z-index: 1;
    }
    
    /* Chat messages */
    .chat-message {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        border-left: 4px solid """ + APP_CONFIG["secondary_color"] + """;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .chat-message:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    
    /* Sidebar moderno */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
        border-right: 1px solid rgba(0,0,0,0.1);
    }
    
    /* Expanders modernos */
    .stExpander > div > div > div > div {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        border: 1px solid rgba(0,0,0,0.1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: """ + APP_CONFIG["secondary_color"] + """;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Selectbox */
    .stSelectbox > div > div > div > div {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        background: white;
    }
    
    /* Notificaciones */
    .success-notification {
        background: linear-gradient(135deg, """ + APP_CONFIG["success_color"] + """ 0%, #059669 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
    }
    
    .warning-notification {
        background: linear-gradient(135deg, """ + APP_CONFIG["warning_color"] + """ 0%, #d97706 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.2);
    }
    
    .error-notification {
        background: linear-gradient(135deg, """ + APP_CONFIG["error_color"] + """ 0%, #dc2626 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.2);
    }
    
    /* Animaciones */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.2rem;
        }
        .main-header p {
            font-size: 1rem;
        }
        .content-container {
            padding: 1.5rem;
        }
        .metric-card {
            padding: 1rem;
        }
    }
    
    /* Scrollbar personalizada */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: """ + APP_CONFIG["secondary_color"] + """;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: """ + APP_CONFIG["primary_color"] + """;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def create_header():
    """Crea el header principal de la aplicación"""
    st.markdown("""
    <div class="main-header fade-in-up">
        <h1>🏦 """ + APP_CONFIG["brand_name"] + """</h1>
        <p>""" + APP_CONFIG["brand_description"] + """</p>
    </div>
    """, unsafe_allow_html=True)

def show_status_info():
    """Muestra información del estado del sistema"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Buscador - con información real
        search_status = show_search_status(st.session_state.vectorstore)
        st.markdown("""
        <div class="metric-card fade-in-up">
            <h4>🔍 Buscador</h4>
            <p>""" + search_status + """</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Botón para actualizar buscador
        if st.button("🔄 Actualizar", use_container_width=True, key="metric_update_search"):
            with st.spinner("Actualizando buscador..."):
                if refresh_vectorstore_cache():
                    st.success("✅ Buscador actualizado")
                    st.rerun()
                else:
                    st.error("❌ Error al actualizar")
    
    with col2:
        # Documentos - con información real y botón
        docs = get_loaded_document_names(DOCUMENTS_DIR)
        st.markdown("""
        <div class="metric-card fade-in-up">
            <h4>📚 Documentos</h4>
            <p>""" + str(len(docs)) + """ archivos</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Botón para ir a documentos
        if st.button("📋 Ver Documentos", use_container_width=True, key="metric_view_docs"):
            st.session_state.main_tab = "Documentos"
            st.rerun()
    
    with col3:
        # Chat - con información real y botón
        messages_count = len(st.session_state.get("messages", []))
        st.markdown("""
        <div class="metric-card fade-in-up">
            <h4>💬 Chat</h4>
            <p>""" + str(messages_count) + """ mensajes</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Botón para ir al chat
        if st.button("💬 Ir al Chat", use_container_width=True, key="metric_go_chat"):
            st.session_state.main_tab = "Chat"
            st.rerun()
    
    # Información adicional del sistema
    st.markdown("---")
    
    # Mostrar detalles del sistema
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Estado del vectorstore
        if st.session_state.vectorstore:
            st.success("✅ Vectorstore: Conectado")
        else:
            st.error("❌ Vectorstore: No disponible")
    
    with col2:
        # Estado del modelo LLM - verificar en session_state
        llm_available = st.session_state.get("llm_model_available", False)
        if llm_available:
            st.success("✅ LLM: Disponible")
        else:
            st.warning("⚠️ IA: No disponible")
    
    with col3:
        # Documentos en el directorio
        docs = get_loaded_document_names(DOCUMENTS_DIR)
        if docs:
            st.info(f"📁 Directorio: {len(docs)} archivos")
        else:
            st.info("📁 Directorio: Vacío")
    
    with col4:
        # Estado del chat
        messages_count = len(st.session_state.get("messages", []))
        if messages_count > 0:
            st.info(f"💬 Chat: {messages_count} mensajes")
        else:
            st.info("💬 Chat: Sin mensajes")

# =============================================================================
# FUNCIONES DEL MOTOR DE BÚSQUEDA
# =============================================================================

def show_search_status(vectorstore):
    """Muestra el estado actual del motor de búsqueda"""
    if vectorstore is None:
        return "❌ No inicializado"
    
    try:
        # Intentar obtener información del vectorstore
        collection = vectorstore._collection
        if collection:
            try:
                count = collection.count()
                if count > 0:
                    return f"✅ Activo ({count} docs)"
                else:
                    return "⚠️ Vacío (0 docs)"
            except Exception:
                return "✅ Conectado"
        else:
            return "⚠️ Parcial"
    except Exception as e:
        # Si hay error específico con Ollama
        if "404" in str(e) or "Ollama" in str(e):
            return "❌ Ollama no disponible"
        else:
            return "❌ Error de conexión"

def refresh_vectorstore_cache():
    """Actualiza la caché del vectorstore"""
    try:
        global vectorstore_cache
        if 'vectorstore_cache' in globals():
            del vectorstore_cache
        return True
    except Exception:
        return False

def get_loaded_document_names(documents_dir):
    """Obtiene la lista de nombres de documentos cargados"""
    try:
        if not documents_dir.exists():
            return []
        files = []
        for file_path in documents_dir.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in APP_CONFIG["supported_formats"]:
                files.append(file_path.name)
        return sorted(files)
    except Exception:
        return []

def initialize_vectorstore_async(documents_dir, chroma_db_dir):
    """Inicializa el vectorstore de forma asíncrona con modo offline"""
    try:
        # Mostrar loader personalizado
        show_custom_loader("🏦 Inicializando base de datos offline...")
        
        documents_dir.mkdir(exist_ok=True)
        chroma_db_dir.mkdir(exist_ok=True)
        
        # Actualizar mensaje del loader
        show_custom_loader("🏦 Cargando modelos de embeddings offline...")
        
        # Configuración para modo offline - usar caché local
        import os
        os.environ['HF_HUB_OFFLINE'] = '1'  # Forzar modo offline
        os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Forzar modo offline
        
        # Intentar cargar embeddings desde caché local
        try:
            embeddings = SentenceTransformerEmbeddings(
                model_name=APP_CONFIG["embedding_model"],
                cache_folder="./embeddings_cache"  # Usar caché local
            )
            
            # Probar el modelo con texto local
            test_embedding = embeddings.embed_query("Prueba offline")
            if not test_embedding or len(test_embedding) == 0:
                raise Exception("Modelo de embeddings no responde")
                
        except Exception as e:
            print(f"⚠️ Error cargando embeddings: {e}")
            print("🔄 Intentando cargar desde caché local...")
            
            # Intentar cargar directamente desde el directorio de caché
            cache_path = Path("./embeddings_cache")
            if cache_path.exists():
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./embeddings_cache')
                    embeddings = SentenceTransformerEmbeddings(
                        model_name=APP_CONFIG["embedding_model"],
                        cache_folder="./embeddings_cache"
                    )
                except Exception as cache_error:
                    print(f"❌ Error con caché local: {cache_error}")
                    # Crear embeddings básicos como fallback
                    return create_basic_vectorstore(documents_dir, chroma_db_dir)
            else:
                print("❌ No se encontró caché local de embeddings")
                return create_basic_vectorstore(documents_dir, chroma_db_dir)
        
        vectorstore = Chroma(
            persist_directory=str(chroma_db_dir),
            embedding_function=embeddings
        )
        
        # Ocultar loader al completar
        hide_custom_loader()
        return vectorstore
    except Exception as e:
        print(f"Error initializing vectorstore: {e}")
        # Ocultar loader en caso de error
        hide_custom_loader()
        return create_basic_vectorstore(documents_dir, chroma_db_dir)

def create_basic_vectorstore(documents_dir, chroma_db_dir):
    """Crea un vectorstore básico para funcionamiento offline sin embeddings"""
    try:
        print("🔄 Creando vectorstore básico offline...")
        
        # Crear un vectorstore simple sin embeddings
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import FakeEmbeddings
        
        # Usar embeddings falsos para funcionamiento básico
        fake_embeddings = FakeEmbeddings(size=384)
        
        vectorstore = Chroma(
            persist_directory=str(chroma_db_dir),
            embedding_function=fake_embeddings
        )
        
        print("✅ Vectorstore básico creado para modo offline")
        return vectorstore
    except Exception as e:
        print(f"❌ Error creando vectorstore básico: {e}")
        return None

def get_ollama_llm():
    """Obtiene la instancia de Ollama LLM optimizada para Mistral con manejo de errores mejorado"""
    # Priorizar Mistral y modelos optimizados
    models_to_try = [
        "mistral",  # Modelo principal optimizado
        "mistral:7b",  # Versión específica de Mistral
        "mistral:latest",  # Última versión de Mistral
        "llama2:7b",  # Fallback a Llama2
        "llama2",  # Fallback genérico
        "codellama:7b"  # Fallback para código
    ]
    
    for model in models_to_try:
        try:
            print(f"🔄 Intentando conectar con modelo: {model}")
            
            # Mostrar loader personalizado
            show_custom_loader(f"🏦 Conectando con {model}...")
            
            # Configuración optimizada para Mistral
            llm_config = {
                "model": model,
                "temperature": 0.1,  # Baja temperatura para respuestas más consistentes
                "top_p": 0.9,  # Control de diversidad
                "top_k": 40,  # Control de tokens
                "repeat_penalty": 1.1,  # Evitar repeticiones
                "num_ctx": 4096,  # Contexto ampliado
            }
            
            llm = Ollama(**llm_config)
            
            # Actualizar mensaje del loader
            show_custom_loader(f"🏦 Probando {model}...")
            
            # Probar una consulta simple para verificar que funciona
            test_response = llm.invoke("Hola, ¿estás listo?")
            
            if test_response and len(test_response.strip()) > 0:
                print(f"✅ Modelo {model} conectado exitosamente")
                print(f"📊 Configuración: temp={llm_config['temperature']}, ctx={llm_config['num_ctx']}")
                
                # Ocultar loader al conectar exitosamente
                hide_custom_loader()
                return llm
            else:
                print(f"⚠️ Modelo {model} respondió vacío")
                continue
                
        except Exception as e:
            print(f"❌ Error con modelo {model}: {str(e)[:100]}...")
            continue
    
    # Si ningún modelo funciona, mostrar información útil
    print("⚠️ No se pudo conectar con ningún modelo de Ollama")
    print("💡 Sugerencias:")
    print("   1. Verifica que Ollama esté ejecutándose: ollama serve")
    print("   2. Descarga Mistral: ollama pull mistral")
    print("   3. Verifica la conexión: ollama list")
    
    # Ocultar loader al fallar
    hide_custom_loader()
    return None

def get_or_create_qa_chain(vectorstore, llm_model):
    """Obtiene o crea la cadena de QA optimizada para Mistral"""
    try:
        if vectorstore is None or llm_model is None:
            return None
        
        # Mostrar loader personalizado
        show_custom_loader("🏦 Configurando cadena de IA...")
        
        # Prompt template optimizado para Mistral
        template = """Eres un asistente inteligente especializado en análisis de documentos. Tu tarea es responder preguntas basándote únicamente en la información proporcionada en el contexto.

CONTEXTO:
{context}

PREGUNTA: {question}

INSTRUCCIONES:
1. Responde de manera clara, concisa y precisa
2. Si la información no está en el contexto, di claramente "No tengo información suficiente para responder esta pregunta"
3. Si la pregunta es ambigua, pide aclaración
4. Usa un tono profesional pero amigable
5. Cita información específica del contexto cuando sea relevante
6. Si hay múltiples fuentes de información, sintetiza la respuesta de manera coherente

RESPUESTA:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Actualizar mensaje del loader
        show_custom_loader("🏦 Configurando buscador...")
        
        # Configuración optimizada para el retriever
        retriever_config = {
            "search_type": "similarity",
            "search_kwargs": {
                "k": APP_CONFIG["retriever_k"]
            }
        }
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_model,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(**retriever_config),
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": False
            },
            return_source_documents=True
        )
        
        # Ocultar loader al completar
        hide_custom_loader()
        return qa_chain
    except Exception as e:
        print(f"Error creating QA chain: {e}")
        # Ocultar loader en caso de error
        hide_custom_loader()
        return None

# =============================================================================
# FUNCIONES DE PROCESAMIENTO DE DOCUMENTOS
# =============================================================================

def load_document_async(file_path):
    """Carga un documento de forma asíncrona con soporte offline"""
    try:
        if file_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() == '.docx':
            loader = Docx2txtLoader(str(file_path))
        elif file_path.suffix.lower() == '.txt':
            # Para archivos de texto, usar encoding UTF-8
            loader = TextLoader(str(file_path), encoding='utf-8')
        else:
            return None
        
        documents = loader.load()
        
        # Procesamiento offline: dividir documentos en chunks más pequeños
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        split_documents = text_splitter.split_documents(documents)
        return split_documents
    except Exception as e:
        print(f"Error loading document {file_path}: {e}")
        return None

def process_uploaded_file(uploaded_file, vectorstore):
    """Procesa un archivo subido y devuelve resultados de análisis"""
    try:
        if uploaded_file.size > APP_CONFIG["max_file_size"]:
            return {"success": False, "error": f"Archivo demasiado grande (máximo {APP_CONFIG['max_file_size'] // (1024*1024)}MB)"}
        
        # Mostrar loader personalizado
        show_custom_loader("🏦 Analizando archivo...")
        
        temp_path = Path(f"./temp_{uploaded_file.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Actualizar mensaje del loader
        show_custom_loader("🏦 Extrayendo contenido...")
        
        documents = load_document_async(temp_path)
        if not documents:
            temp_path.unlink(missing_ok=True)
            hide_custom_loader()
            return {"success": False, "error": "No se pudo cargar el documento"}
        
        content = " ".join([doc.page_content for doc in documents])
        lines = len(content.split('\n'))
        words = len(content.split())
        characters = len(content)
        
        existing_docs = get_loaded_document_names(DOCUMENTS_DIR)
        is_duplicate = uploaded_file.name in existing_docs
        
        temp_path.unlink(missing_ok=True)
        
        # Ocultar loader al completar
        hide_custom_loader()
        
        return {
            "success": True,
            "is_duplicate": is_duplicate,
            "duplicate_reason": "Archivo con el mismo nombre ya existe" if is_duplicate else None,
            "existing_file": uploaded_file.name if is_duplicate else None,
            "file_stats": {
                "lines": lines,
                "words": words,
                "characters": characters
            },
            "similar_docs": [],
            "has_changes": False,
            "comparison_results": [],
            "auto_comment": f"Archivo {uploaded_file.name} analizado. Contiene {words} palabras y {lines} líneas."
        }
    except Exception as e:
        hide_custom_loader()
        return {"success": False, "error": str(e)}

def save_uploaded_file(uploaded_file, vectorstore):
    """Guarda un archivo subido en el directorio de documentos"""
    try:
        # Mostrar loader personalizado
        show_custom_loader("🏦 Guardando archivo...")
        
        file_path = DOCUMENTS_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Actualizar mensaje del loader
        show_custom_loader("🏦 Procesando documento...")
        
        documents = load_document_async(file_path)
        if documents and vectorstore:
            vectorstore.add_documents(documents)
        
        # Ocultar loader al completar
        hide_custom_loader()
        
        return {
            "success": True,
            "filename": uploaded_file.name,
            "file_size": uploaded_file.size,
            "documents_added": len(documents) if documents else 0
        }
    except Exception as e:
        hide_custom_loader()
        return {"success": False, "error": str(e)}

# =============================================================================
# FUNCIONES DE GESTIÓN DE COMENTARIOS
# =============================================================================

def add_comment_to_db(comment_text, vectorstore):
    """Agrega un comentario a la base de datos"""
    try:
        if not comment_text.strip():
            return None
        
        doc = Document(
            page_content=comment_text,
            metadata={
                "source": "Comentario del usuario",
                "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )
        
        if vectorstore:
            vectorstore.add_documents([doc])
        
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"Error adding comment: {e}")
        return None

def display_saved_comments(vectorstore):
    """Muestra los comentarios guardados"""
    st.markdown("""
    <div class="success-notification fade-in-up">
        <h4>📋 Comentarios del Sistema</h4>
        <p>Los comentarios se muestran automáticamente en el chat cuando son relevantes para las consultas.</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# FUNCIONES DE GESTIÓN DE OLLAMA
# =============================================================================

def check_ollama_status():
    """Verifica el estado de Ollama y los modelos disponibles"""
    try:
        import subprocess
        import json
        
        # Mostrar loader personalizado
        show_custom_loader("🏦 Verificando estado de Ollama...")
        
        # Verificar si Ollama está ejecutándose
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                models = result.stdout.strip().split('\n')[1:]  # Saltar header
                available_models = []
                for model in models:
                    if model.strip():
                        model_name = model.split()[0]
                        available_models.append(model_name)
                
                # Ocultar loader al completar
                hide_custom_loader()
                return {
                    "status": "running",
                    "models": available_models,
                    "has_mistral": any("mistral" in model.lower() for model in available_models)
                }
            else:
                hide_custom_loader()
                return {"status": "error", "message": "Ollama no responde correctamente"}
        except subprocess.TimeoutExpired:
            hide_custom_loader()
            return {"status": "timeout", "message": "Ollama no responde (timeout)"}
        except FileNotFoundError:
            hide_custom_loader()
            return {"status": "not_installed", "message": "Ollama no está instalado"}
        except Exception as e:
            hide_custom_loader()
            return {"status": "error", "message": str(e)}
            
    except Exception as e:
        hide_custom_loader()
        return {"status": "error", "message": f"Error verificando Ollama: {str(e)}"}

def suggest_ollama_setup():
    """Sugiere la configuración de Ollama basada en el estado actual"""
    status = check_ollama_status()
    
    if status["status"] == "not_installed":
        return {
            "title": "🔧 Ollama no está instalado",
            "message": "Necesitas instalar Ollama para usar el chat inteligente.",
            "steps": [
                "1. **Windows**: `winget install Ollama.Ollama` o descarga desde https://ollama.ai",
                "2. **macOS**: `brew install ollama`",
                "3. **Linux**: `curl -fsSL https://ollama.ai/install.sh | sh`",
                "4. Reinicia tu terminal después de la instalación"
            ]
        }
    elif status["status"] == "timeout" or status["status"] == "error":
        return {
            "title": "⚠️ Ollama no está ejecutándose",
            "message": "Ollama está instalado pero no está ejecutándose.",
            "steps": [
                "1. Abre una nueva terminal",
                "2. Ejecuta: `ollama serve`",
                "3. Mantén esa terminal abierta",
                "4. Regresa aquí y recarga la página"
            ]
        }
    elif status["status"] == "running" and not status.get("has_mistral", False):
        return {
            "title": "📥 Descarga el modelo Mistral",
            "message": "Ollama está funcionando pero necesitas descargar Mistral.",
            "steps": [
                "1. Abre una terminal",
                "2. Ejecuta: `ollama pull mistral`",
                "3. Espera a que termine la descarga",
                "4. Regresa aquí y recarga la página"
            ]
        }
    else:
        return {
            "title": "✅ Ollama está listo",
            "message": "Ollama está funcionando correctamente con Mistral.",
            "steps": []
        }

# =============================================================================
# FUNCIONES DE OLLAMA OPTIMIZADAS
# =============================================================================

# =============================================================================
# FUNCIONES DE INTERFAZ
# =============================================================================

def clear_upload_interface():
    """Limpia la interfaz de subida"""
    if "file_uploader_key" in st.session_state:
        st.session_state.file_uploader_key += 1

def clear_comment_interface():
    """Limpia la interfaz de comentarios"""
    st.session_state.comment_text_value = ""
    st.session_state.show_comment_input_area = False
    st.session_state.show_post_save_return_option = False
    st.session_state.editing_comment = False

def clear_chat_history():
    """Limpia el historial del chat"""
    st.session_state.messages = []

def clear_and_regenerate_database():
    """Limpia y regenera la base de datos"""
    try:
        if st.session_state.vectorstore:
            st.session_state.vectorstore = None
        
        if CHROMA_DB_DIR.exists():
            shutil.rmtree(CHROMA_DB_DIR)
        
        st.session_state.vectorstore = initialize_vectorstore_async(DOCUMENTS_DIR, CHROMA_DB_DIR)
        st.success("Base de datos limpiada y regenerada.")
    except Exception as e:
        st.error(f"Error al limpiar base de datos: {e}")

def show_modal_notification(title, message, success=True):
    """Muestra una notificación modal con el nuevo diseño"""
    if success:
        st.markdown(f"""
        <div class="success-notification fade-in-up">
            <h4>✅ {title}</h4>
            <p>{message}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="error-notification fade-in-up">
            <h4>❌ {title}</h4>
            <p>{message}</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# FUNCIONES DE BÚSQUEDA Y ANÁLISIS
# =============================================================================

def get_search_suggestions(query):
    """Obtiene sugerencias de búsqueda basadas en la consulta"""
    suggestions = []
    query_lower = query.lower()
    
    if "documento" in query_lower or "archivo" in query_lower:
        suggestions.append("Buscar por nombre de archivo específico")
    if "fecha" in query_lower or "cuando" in query_lower:
        suggestions.append("Buscar por fecha o período de tiempo")
    if "comentario" in query_lower:
        suggestions.append("Revisar comentarios guardados")
    
    return suggestions[:3]

def show_search_stats(query, source_docs):
    """Muestra estadísticas de búsqueda"""
    if not source_docs:
        return None
    
    total_sources = len(source_docs)
    unique_sources = len(set(doc.metadata.get('source', 'Unknown') for doc in source_docs))
    
    return f"""
    **📊 Estadísticas de búsqueda:**
    - Fuentes encontradas: {total_sources}
    - Documentos únicos: {unique_sources}
    - Consulta: "{query}"
    """

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Función principal de la aplicación"""
    setup_page_config()

    # Inicializar variables de sesión
    session_defaults = {
        "vectorstore": None,
        "messages": [],
        "show_comment_input_area": False,
        "comment_text_value": "",
        "show_saved_comments_section": False,
        "show_post_save_return_option": False,
        "show_chat_interface": False,  # Cambiado a False por defecto
        "show_upload_file_area": False,
        "main_tab": "Principal",  # Página principal por defecto
        "search_query": "",  # Nueva variable para búsqueda
        "show_documents": False  # Nueva variable para mostrar documentos
    }
    for k, v in session_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Header principal
    create_header()
    
    # Sidebar para navegación y funcionalidades
    with st.sidebar:
        st.header("📋 Navegación")
        
        # Barra de búsqueda en el sidebar
        st.subheader("🔍 Buscar Documentos")
        search_query = st.text_input(
            "Buscar en documentos...",
            value=st.session_state.get("search_query", ""),
            placeholder="Escribe para buscar...",
            key="sidebar_search"
        )
        
        if search_query:
            st.session_state.search_query = search_query
            st.session_state.show_documents = True
            st.session_state.main_tab = "Búsqueda"
        
        st.divider()
        
        # Gestión de documentos
        st.subheader("📚 Documentos")
        
        # Botón para subir archivos
        if st.button("📁 Subir archivo", use_container_width=True, key="sidebar_upload"):
            st.session_state.main_tab = "Subir archivo"
            st.rerun()
        
        # Botón para ver documentos
        if st.button("📋 Ver documentos", use_container_width=True, key="sidebar_documents"):
            st.session_state.show_documents = True
            st.session_state.main_tab = "Documentos"
            st.rerun()
        
        st.divider()
        
        # Sistema de comentarios
        st.subheader("💬 Comentarios")
        
        # Botón para ver comentarios
        if st.button("👁️ Ver comentarios", use_container_width=True, key="sidebar_view_comments"):
            st.session_state.main_tab = "Comentarios"
            st.rerun()
        
        # Botón para agregar comentarios
        if st.button("✏️ Agregar comentario", use_container_width=True, key="sidebar_add_comment"):
            st.session_state.main_tab = "Agregar Comentario"
            st.rerun()
        
        # Botón para limpiar comentarios
        if st.button("🧹 Limpiar comentarios", use_container_width=True, key="sidebar_clear_comments"):
            st.session_state.messages = []
            st.success("Comentarios limpiados.")
            st.rerun()
        
        st.divider()
        
        # Utilidades
        st.subheader("🛠️ Utilidades")
        
        # Botón para limpiar chat
        if st.button("🧹 Limpiar chat", use_container_width=True, key="sidebar_clear_chat"):
            clear_chat_history()
            st.success("Historial de chat limpiado.")
            st.rerun()
        
        # Botón para ir al inicio
        if st.button("🏠 Inicio", use_container_width=True, key="sidebar_home"):
            st.session_state.main_tab = "Principal"
            st.session_state.show_chat_interface = False
            st.session_state.show_documents = False
            st.rerun()
        
        # Sección de ayuda para Ollama (solo si no está disponible)
        if not st.session_state.get("ollama_available", False):
            st.divider()
            if st.button("🔧 Configurar Ollama", use_container_width=True, key="sidebar_ollama_help"):
                st.session_state.main_tab = "Configurar Ollama"
                st.rerun()

    # Inicialización de componentes optimizada para modo offline
    try:
        # Verificar estado de Ollama primero
        ollama_status = check_ollama_status()
        st.session_state.ollama_status = ollama_status
        
        # Verificar modelos offline
        offline_models_available = check_offline_models()
        if not offline_models_available:
            print("⚠️ Modelos offline no disponibles, intentando descargar...")
            download_offline_models()
        
        # Inicializar vectorstore con modelos offline
        st.session_state.vectorstore = initialize_vectorstore_async(DOCUMENTS_DIR, CHROMA_DB_DIR)
        
        # Inicializar LLM offline solo si Ollama está disponible
        llm_model = None
        if ollama_status["status"] == "running":
            llm_model = get_offline_llm()  # Usar función offline
            if llm_model is not None:
                st.session_state.ollama_available = True
                st.session_state.llm_model_available = True
                st.session_state.offline_mode = True
            else:
                st.session_state.ollama_available = False
                st.session_state.llm_model_available = False
                st.session_state.offline_mode = False
        else:
            st.session_state.ollama_available = False
            st.session_state.llm_model_available = False
            st.session_state.offline_mode = False
            
    except Exception as e:
        print(f"Error al inicializar componentes offline: {e}")
        st.session_state.vectorstore = None
        llm_model = None
        st.session_state.ollama_available = False
        st.session_state.llm_model_available = False
        st.session_state.offline_mode = False

    # Solo crear qa_chain si tenemos llm_model
    qa_chain = None
    if llm_model is not None:
        qa_chain = get_or_create_qa_chain(st.session_state.vectorstore, llm_model)
        if qa_chain is None:
            st.session_state.llm_model_available = False
    else:
        st.session_state.llm_model_available = False

    # =============================================================================
    # INTERFAZ PRINCIPAL SIMPLIFICADA
    # =============================================================================
    
    if st.session_state.main_tab == "Principal":
        # Estado del servicio en la esquina superior izquierda
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown("### 🔧 Estado del Servicio")
            
            # Estado de Ollama
            if st.session_state.get("ollama_available", False):
                st.success("✅ Ollama: Activo (Offline)")
            else:
                st.error("❌ Ollama: Inactivo")
            
            # Estado del vectorstore
            if st.session_state.vectorstore:
                st.success("✅ Base de datos: Activa (Offline)")
            else:
                st.error("❌ Base de datos: Inactiva")
            
            # Estado del modelo LLM
            if st.session_state.get("llm_model_available", False):
                st.success("✅ IA: Disponible (Offline)")
            else:
                st.warning("⚠️ IA: No disponible")
            
            # Estado del modo offline
            if st.session_state.get("offline_mode", False):
                st.success("✅ Modo: Offline")
            else:
                st.info("ℹ️ Modo: Requiere conexión")
        
        with col2:
            st.markdown("### 🏦 Banco de Corrientes")
            st.markdown("**Asistente Inteligente Offline**")
            st.markdown("*Funciona sin conexión a internet*")
            
            # Estadísticas rápidas
            docs = get_loaded_document_names(DOCUMENTS_DIR)
            messages_count = len(st.session_state.get("messages", []))
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("📚 Documentos", len(docs))
            with col_b:
                st.metric("💬 Mensajes", messages_count)
        
        with col3:
            st.markdown("### 🚀 Acciones Rápidas")
            
            # Botón principal para Chat Inteligente
            if st.button("💬 Chat Inteligente", use_container_width=True, type="primary", key="main_chat_btn"):
                st.session_state.show_chat_interface = True
                st.session_state.main_tab = "Chat"
                st.rerun()
            
            # Botón para ver documentos
            if st.button("📋 Ver Documentos", use_container_width=True, key="main_docs_btn"):
                st.session_state.show_documents = True
                st.session_state.main_tab = "Documentos"
                st.rerun()
        
        st.divider()
        
        # Información adicional
        st.markdown("""
        <div class="success-notification fade-in-up">
            <h4>🚀 Funcionalidades Offline Disponibles</h4>
            <ul>
                <li><strong>💬 Chat Inteligente:</strong> Haz preguntas sobre tus documentos (sin internet)</li>
                <li><strong>📁 Subir Archivo:</strong> Agrega nuevos documentos al sistema (PDF, DOCX, TXT)</li>
                <li><strong>📚 Ver Documentos:</strong> Explora los documentos disponibles</li>
                <li><strong>💬 Comentarios:</strong> Gestiona comentarios y notas</li>
                <li><strong>🔍 Búsqueda:</strong> Búsqueda semántica en documentos</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar información sobre Ollama si no está disponible
        if not st.session_state.get("ollama_available", False):
            st.markdown("""
            <div class="warning-notification fade-in-up">
                <h4>💡 Configuración para Modo Offline</h4>
                <p>Para usar el <strong>Chat Inteligente Offline</strong>, necesitas instalar Ollama:</p>
                
                <details>
                <summary><strong>📋 Instrucciones de Instalación Offline</strong></summary>
                
                <h5>1. Instalar Ollama (una sola vez):</h5>
                <ul>
                <li><strong>Windows:</strong> <code>winget install Ollama.Ollama</code></li>
                <li><strong>macOS:</strong> <code>brew install ollama</code></li>
                <li><strong>Linux:</strong> <code>curl -fsSL https://ollama.ai/install.sh | sh</code></li>
                </ul>
                
                <h5>2. Iniciar Ollama:</h5>
                <code>ollama serve</code>
                
                <h5>3. Descargar modelo (una sola vez):</h5>
                <code>ollama pull mistral</code>
                
                <h5>4. ¡Listo! Ya funciona offline</h5>
                </details>
                
                <p><em>💡 <strong>Ventaja:</strong> Una vez configurado, funciona completamente sin internet.</em></p>
                </div>
            """, unsafe_allow_html=True)

    # =============================================================================
    # CHAT INTELIGENTE - INTERFAZ INTEGRADA
    # =============================================================================
    
    elif st.session_state.main_tab == "Chat":
        # Header del chat
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 💬 Chat Inteligente")
            st.markdown("*Haz preguntas sobre tus documentos y comentarios*")
        
        with col2:
            if st.button("🏠 Volver al Inicio", use_container_width=True, key="chat_home"):
                st.session_state.main_tab = "Principal"
                st.session_state.show_chat_interface = False
                st.rerun()
        
        st.divider()
        
        # Verificar si el modelo está disponible
        if llm_model is None or qa_chain is None:
            # Usar la nueva función de sugerencias
            setup_info = suggest_ollama_setup()
            
            st.markdown(f"""
            <div class="warning-notification fade-in-up">
                <h4>{setup_info['title']}</h4>
                <p>{setup_info['message']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if setup_info['steps']:
                st.markdown("### 🔧 Instrucciones de Configuración:")
                for step in setup_info['steps']:
                    st.markdown(step)
                
                # Botón para verificar estado
                if st.button("🔄 Verificar Estado de Ollama", key="check_ollama_status"):
                    st.rerun()
            
            # Mostrar estado actual de Ollama
            if hasattr(st.session_state, 'ollama_status'):
                status = st.session_state.ollama_status
                st.markdown("### 📊 Estado Actual de Ollama:")
                
                if status["status"] == "running":
                    st.success(f"✅ Ollama está ejecutándose")
                    if status.get("models"):
                        st.info(f"📦 Modelos disponibles: {', '.join(status['models'][:5])}")
                        if len(status['models']) > 5:
                            st.info(f"... y {len(status['models']) - 5} más")
                    else:
                        st.warning("⚠️ No hay modelos descargados")
                elif status["status"] == "not_installed":
                    st.error("❌ Ollama no está instalado")
                elif status["status"] == "timeout":
                    st.error("⏰ Ollama no responde (timeout)")
                else:
                    st.error(f"❌ Error: {status.get('message', 'Desconocido')}")
            
            # Mostrar documentos disponibles para consulta manual
            docs = get_loaded_document_names(DOCUMENTS_DIR)
            if docs:
                st.markdown("### 📚 Documentos Disponibles para Consulta Manual:")
                for doc in docs:
                    with st.expander(f"📄 {doc}"):
                        doc_path = DOCUMENTS_DIR / doc
                        try:
                            loaded = load_document_async(doc_path)
                            if loaded:
                                doc_content = "\n".join([d.page_content for d in loaded])
                                st.text(doc_content[:500] + "..." if len(doc_content) > 500 else doc_content)
                        except Exception:
                            st.text("(No se pudo cargar el contenido)")
            else:
                st.markdown("""
                <div class="success-notification fade-in-up">
                    <h4>📁 No hay documentos</h4>
                    <p>Sube algunos documentos para comenzar a usar el sistema.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Chat normal cuando el modelo está disponible
            if not st.session_state.messages:
                st.markdown("""
                <div class="success-notification fade-in-up">
                    <h4>👋 ¡Bienvenido al Chat Inteligente!</h4>
                    <p>Haz una pregunta sobre tus documentos o comentarios. Estoy aquí para ayudarte.</p>
                </div>
                """, unsafe_allow_html=True)

            # Mostrar mensajes del chat
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(f"""
                    <div class="chat-message fade-in-up">
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)

            # Input del chat
            prompt = st.chat_input("Escribe tu pregunta aquí...", disabled=st.session_state.get("input_disabled", False))
            if prompt:
                st.session_state.messages = []
                st.session_state.input_disabled = True
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(f"""
                    <div class="chat-message fade-in-up">
                        {prompt}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Mostrar loader personalizado del Banco de Corrientes
                show_custom_loader("🏦 Banco de Corrientes está procesando tu consulta...")
                
                try:
                    if not qa_chain:
                        hide_custom_loader()
                        st.error("El motor de consultas no está listo. Intente recargar la página.")
                        st.session_state.messages.append({"role": "assistant", "content": "El agente no está listo para responder. Por favor, recargue la página."})
                        st.session_state.input_disabled = False
                        return

                    result = qa_chain.invoke({"query": prompt})
                    response_text = result.get("result", "")
                    source_docs = result.get("source_documents", [])

                    response_lower = response_text.strip().lower()
                    general_responses = [
                        "no lo sé", "no lo se", "no sé", "no se",
                        "no tengo información", "no hay información", "no encuentro información",
                        "no conozco la respuesta", "no tengo datos", "no hay datos",
                        "no puedo encontrar", "no puedo localizar", "no tengo acceso",
                        "no está disponible", "no se encuentra", "no existe información"
                    ]
                    query_words = [word.lower() for word in prompt.split() if len(word) > 2]
                    contains_specific_words = any(word in response_lower for word in query_words)
                    is_too_general = (
                        len(response_text.strip()) < 50 or
                        response_lower.count("documento") > 3 or
                        response_lower.count("información") > 2 or
                        (not contains_specific_words and len(query_words) > 0)
                    )
                    has_useful_info = (
                        response_text.strip() and
                        not any(general in response_lower for general in general_responses) and
                        not is_too_general and
                        contains_specific_words
                    )
                    if not has_useful_info:
                        response_text = "no conozco la respuesta, no tengo informacion disponible para responder la pregunta"
                        source_docs = []
                        formatted_response = response_text
                        suggestions = get_search_suggestions(prompt)
                        if suggestions:
                            formatted_response += f"\n\n💡 **Sugerencias de búsqueda:**\n"
                            for suggestion in suggestions:
                                formatted_response += f"• {suggestion}\n"
                            st.session_state.messages = []
                    else:
                        formatted_response = response_text

                    GENERIC_NO_UNDERSTAND = [
                        "no entiendo la pregunta",
                        "por favor proporciona más contexto",
                        "rephraze la pregunta",
                        "reformula la pregunta",
                        "necesito más información",
                        "puedes ser más específico",
                        "no está claro qué buscas"
                    ]
                    if any(kw in response_lower for kw in GENERIC_NO_UNDERSTAND):
                        source_docs = []
                        formatted_response = response_text

                    # Mostrar fuentes como expansibles
                    if source_docs and response_text.strip() and response_text != "no conozco la respuesta, no tengo informacion disponible para responder la pregunta" and "no conozco la respuesta" not in response_text.lower():
                        fuentes_utilizadas = []
                        for i, doc in enumerate(source_docs):
                            source_name = doc.metadata.get('source', 'Desconocido')
                            page_label = doc.metadata.get('page')
                            creation_date = doc.metadata.get('creation_date', 'N/A')
                            source_detail = f"Fuente {i+1}: "
                            if source_name != "Comentario del usuario":
                                try:
                                    file_name = Path(source_name).name
                                    file_path = Path(source_name)
                                    if file_path.is_absolute():
                                        source_detail += f"`{file_name}` (Ubicación: {file_path})"
                                    else:
                                        source_detail += f"`{file_name}` (Ubicación: documents/{file_name})"
                                except Exception:
                                    file_name = str(source_name)
                                    source_detail += f"`{file_name}` (Ubicación: documents/{file_name})"
                                if page_label is not None:
                                    try:
                                        page_num = int(page_label) + 1
                                    except Exception:
                                        page_num = page_label
                                    source_detail += f", pág. {page_num}"
                            else:
                                source_detail += f"Comentario (fecha: {creation_date})"
                            query_variations_found = doc.metadata.get('query_variations_found', [])
                            if query_variations_found:
                                source_detail += f" | Palabras encontradas: {', '.join(query_variations_found[:3])}"
                            content_preview = doc.page_content.strip()
                            if len(content_preview) > 300:
                                content_preview = content_preview[:300] + "..."
                            fuentes_utilizadas.append((source_detail, content_preview))
                        # Mostrar cada fuente como botón expansible
                        full_response_with_sources = formatted_response
                        st.markdown(full_response_with_sources, unsafe_allow_html=True)
                        st.markdown("#### Fuentes encontradas")
                        for idx, (source_detail, content_preview) in enumerate(fuentes_utilizadas):
                            with st.expander(source_detail):
                                st.markdown(f"<blockquote style='color:var(--text-secondary);'>{content_preview}</blockquote>", unsafe_allow_html=True)
                        # Mostrar estadísticas de búsqueda si hay fuentes
                        if source_docs and response_text.strip() and response_text != "no conozco la respuesta, no tengo informacion disponible para responder la pregunta":
                            search_stats = show_search_stats(prompt, source_docs)
                            if search_stats:
                                st.markdown(f"---\n{search_stats}", unsafe_allow_html=True)
                    else:
                        with st.chat_message("assistant"):
                            st.markdown(formatted_response, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                except Exception as e:
                    print(f"Error en consulta: {e}")
                    error_message = "Ocurrió un error al procesar la consulta. Intente de nuevo o recargue la página."
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    with st.chat_message("assistant"):
                        st.error(error_message)
                finally:
                    # Ocultar loader personalizado
                    hide_custom_loader()
                    st.session_state.input_disabled = False
                    st.rerun()

    # =============================================================================
    # SECCIÓN DE COMENTARIOS
    # =============================================================================
    
    elif st.session_state.main_tab == "Comentarios":
        st.markdown("### 💬 Comentarios del Sistema")
        st.markdown("*Revisa todos los comentarios guardados*")
        
        if st.button("🏠 Volver al Inicio", use_container_width=True, key="comments_home"):
            st.session_state.main_tab = "Principal"
            st.rerun()
        
        st.divider()
        
        # Mostrar comentarios guardados
        display_saved_comments(st.session_state.vectorstore)
    
    elif st.session_state.main_tab == "Agregar Comentario":
        st.markdown("### ✏️ Agregar Comentario")
        st.markdown("*Crea un nuevo comentario o nota*")
        
        if st.button("🏠 Volver al Inicio", use_container_width=True, key="add_comment_home"):
            st.session_state.main_tab = "Principal"
            st.rerun()
        
        st.divider()
        
        # Formulario para agregar comentario
        comment_text = st.text_area(
            "Escribe tu comentario:",
            value=st.session_state.get("comment_text_value", ""),
            height=150,
            placeholder="Escribe aquí tu comentario o nota..."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Guardar Comentario", type="primary", use_container_width=True):
                if comment_text.strip():
                    creation_date = add_comment_to_db(comment_text, st.session_state.vectorstore)
                    if creation_date:
                        st.success(f"✅ Comentario guardado exitosamente el {creation_date}")
                        st.session_state.comment_text_value = ""
                        st.rerun()
                    else:
                        st.error("❌ Error al guardar el comentario")
                else:
                    st.warning("⚠️ El comentario no puede estar vacío")
        
        with col2:
            if st.button("🔄 Limpiar", use_container_width=True):
                st.session_state.comment_text_value = ""
                st.rerun()
    
    elif st.session_state.main_tab == "Búsqueda":
        st.markdown("### 🔍 Resultados de Búsqueda")
        st.markdown(f"*Buscando: '{st.session_state.get('search_query', '')}'*")
        
        if st.button("🏠 Volver al Inicio", use_container_width=True, key="search_home"):
            st.session_state.main_tab = "Principal"
            st.session_state.search_query = ""
            st.rerun()
        
        st.divider()
        
        # Realizar búsqueda en documentos
        search_query = st.session_state.get("search_query", "")
        if search_query and st.session_state.vectorstore:
            try:
                # Buscar en el vectorstore
                results = st.session_state.vectorstore.similarity_search(search_query, k=5)
                
                if results:
                    st.markdown(f"**Encontrados {len(results)} resultados:**")
                    
                    for i, doc in enumerate(results, 1):
                        with st.expander(f"📄 Resultado {i}"):
                            source_name = doc.metadata.get('source', 'Desconocido')
                            st.markdown(f"**Fuente:** {source_name}")
                            st.markdown(f"**Contenido:**")
                            st.markdown(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                else:
                    st.info("No se encontraron resultados para tu búsqueda.")
                    
            except Exception as e:
                st.error(f"Error en la búsqueda: {e}")
        else:
            st.info("Ingresa un término de búsqueda en la barra lateral.")

    # =============================================================================
    # OTRAS SECCIONES
    # =============================================================================

    elif st.session_state.main_tab == "Utilidades":
        st.markdown("""
        <div class="content-container fade-in-up">
            <h3>🛠️ Utilidades del Sistema</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Botón para volver al inicio
        if st.button("🏠 Volver al Inicio", use_container_width=True, key="util_home"):
            st.session_state.main_tab = "Principal"
            st.rerun()
        
        st.divider()
        
        # Opciones de utilidades
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧹 Limpieza del Sistema")
            if st.button("🧹 Limpiar Historial del Chat", use_container_width=True, key="util_clear_chat"):
                clear_chat_history()
                st.success("✅ Historial del chat limpiado exitosamente")
                st.rerun()
            
            st.markdown("### 📊 Estado del Sistema")
            docs = get_loaded_document_names(DOCUMENTS_DIR)
            st.info(f"📚 Documentos cargados: {len(docs)}")
            st.info(f"💬 Mensajes en chat: {len(st.session_state.get('messages', []))}")
        
        with col2:
            st.markdown("### 🗑️ Gestión de Base de Datos")
            if st.button("🗑️ Limpiar Base de Datos", use_container_width=True, key="util_clear_db"):
                st.session_state.main_tab = "Limpiar base de datos"
                st.rerun()
            
            st.markdown("### 🔄 Mantenimiento")
            if st.button("🔄 Actualizar Buscador", use_container_width=True, key="util_update_search"):
                with st.spinner("Actualizando buscador..."):
                    if refresh_vectorstore_cache():
                        st.success("✅ Buscador actualizado exitosamente")
                        st.rerun()
                    else:
                        st.error("❌ Error al actualizar el buscador")

    elif st.session_state.main_tab == "Configurar Ollama":
        st.markdown("""
        <div class="content-container fade-in-up">
            <h3>🔧 Configuración de Ollama</h3>
            <p>Guía paso a paso para configurar Ollama con Mistral y habilitar el chat inteligente.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Botón para volver al inicio
        if st.button("🏠 Volver al Inicio", use_container_width=True, key="ollama_home"):
            st.session_state.main_tab = "Principal"
            st.rerun()
        
        st.divider()
        
        # Estado actual mejorado
        st.markdown("### 📊 Estado Actual de Ollama")
        
        if hasattr(st.session_state, 'ollama_status'):
            status = st.session_state.ollama_status
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if status["status"] == "running":
                    st.success("✅ Ollama: Ejecutándose")
                elif status["status"] == "not_installed":
                    st.error("❌ Ollama: No instalado")
                elif status["status"] == "timeout":
                    st.error("⏰ Ollama: Timeout")
                else:
                    st.error(f"❌ Ollama: Error")
            
            with col2:
                if status.get("has_mistral", False):
                    st.success("✅ Mistral: Disponible")
                elif status["status"] == "running":
                    st.warning("⚠️ Mistral: No descargado")
                else:
                    st.info("ℹ️ Mistral: No verificado")
            
            with col3:
                if st.session_state.get("llm_model_available", False):
                    st.success("✅ Chat: Habilitado")
                else:
                    st.warning("⚠️ Chat: Deshabilitado")
            
            # Mostrar modelos disponibles
            if status.get("models"):
                st.markdown("#### 📦 Modelos Disponibles:")
                for model in status["models"]:
                    st.code(model)
        
        st.divider()
        
        # Instrucciones específicas para Mistral
        st.markdown("## 🚀 Configuración Optimizada para Mistral")
        
        setup_info = suggest_ollama_setup()
        if setup_info['steps']:
            st.markdown(f"### {setup_info['title']}")
            st.info(setup_info['message'])
            
            st.markdown("#### 📋 Pasos a seguir:")
            for i, step in enumerate(setup_info['steps'], 1):
                st.markdown(f"{i}. {step}")
        
        # Instrucciones detalladas
        with st.expander("📖 Instrucciones Detalladas"):
            st.markdown("""
            ### 1. Instalar Ollama
            
            **Windows:**
            ```bash
            winget install Ollama.Ollama
            ```
            
            **macOS:**
            ```bash
            brew install ollama
            ```
            
            **Linux:**
            ```bash
            curl -fsSL https://ollama.ai/install.sh | sh
            ```
            
            ### 2. Iniciar Ollama
            
            Abre una terminal y ejecuta:
            ```bash
            ollama serve
            ```
            
            ### 3. Descargar Mistral (Recomendado)
            
            En otra terminal, ejecuta:
            ```bash
            # Descargar Mistral (modelo principal)
            ollama pull mistral
            
            # Verificar descarga
            ollama list
            ```
            
            ### 4. Modelos Alternativos
            
            Si prefieres otros modelos:
            ```bash
            # Llama2 (más pequeño)
            ollama pull llama2:7b
            
            # Code Llama (para código)
            ollama pull codellama:7b
            ```
            
            ### 5. Verificar Instalación
            
            ```bash
            # Listar modelos
            ollama list
            
            # Probar Mistral
            ollama run mistral "Hola, ¿cómo estás?"
            ```
            """)
        
        # Botones de acción
        st.divider()
        st.markdown("### 🔄 Acciones")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Verificar Estado", use_container_width=True, key="check_status"):
                st.rerun()
        
        with col2:
            if st.button("🧪 Probar Conexión", use_container_width=True, key="test_connection"):
                try:
                    test_llm = get_ollama_llm()
                    if test_llm:
                        st.success("✅ ¡Conexión exitosa! Ollama está funcionando correctamente.")
                        st.session_state.ollama_available = True
                        st.session_state.llm_model_available = True
                        st.rerun()
                    else:
                        st.error("❌ No se pudo conectar con Ollama. Verifica la instalación.")
                except Exception as e:
                    st.error(f"❌ Error al probar conexión: {str(e)}")
        
        # Información adicional
        st.divider()
        st.markdown("### 💡 Información Adicional")
        
        st.info("""
        **¿Por qué Mistral?**
        - 🚀 **Rendimiento**: Mistral es más rápido y eficiente que Llama2
        - 🧠 **Calidad**: Proporciona respuestas más precisas y contextuales
        - 💾 **Memoria**: Usa menos RAM que modelos más grandes
        - 🌍 **Multilingüe**: Excelente soporte para español
        
        **Requisitos del Sistema:**
        - Mínimo 8GB RAM para Mistral
        - Conexión a internet para descargar el modelo
        - Terminal/Command Prompt accesible
        """)

    elif st.session_state.main_tab == "Documentos":
        st.markdown("""
        <div class="content-container fade-in-up">
            <h3>📚 Documentos Disponibles</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Botón para volver al inicio
        if st.button("🏠 Volver al Inicio", use_container_width=True, key="docs_home"):
            st.session_state.main_tab = "Principal"
            st.rerun()
        
        st.divider()
        
        docs = get_loaded_document_names(DOCUMENTS_DIR)
        if docs:
            for doc in docs:
                with st.expander(f"📄 {doc}"):
                    doc_path = DOCUMENTS_DIR / doc
                    doc_content = ""
                    try:
                        loaded = load_document_async(doc_path)
                        if loaded:
                            doc_content = "\n".join([d.page_content for d in loaded])
                    except Exception:
                        doc_content = "(No se pudo cargar el contenido)"
                    st.markdown(f"**Vista previa:**")
                    st.text(doc_content[:1000] + "..." if len(doc_content) > 1000 else doc_content)
        else:
            st.markdown("""
            <div class="warning-notification fade-in-up">
                <h4>📚 No hay documentos</h4>
                <p>Sube algunos documentos para comenzar a usar """ + APP_CONFIG["brand_name"] + """.</p>
            </div>
            """, unsafe_allow_html=True)

    elif st.session_state.main_tab == "Subir archivo":
        st.markdown("""
        <div class="content-container fade-in-up">
            <h3>📁 Subir Archivo</h3>
            <p>Sube un archivo PDF o DOCX para agregarlo a tu base de documentos.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Botón para volver al inicio
        if st.button("🏠 Volver al Inicio", use_container_width=True, key="upload_home"):
            st.session_state.main_tab = "Principal"
            st.rerun()
        
        st.divider()
        
        with st.expander("ℹ️ Información sobre el proceso offline"):
            st.markdown("""
            **Proceso de análisis offline:**
            1. **Validación**: Se verifica el formato y tamaño del archivo
            2. **Detección de duplicados**: Se verifica si el archivo ya existe
            3. **Extracción**: Se extrae el contenido del documento (sin internet)
            4. **Procesamiento**: Se divide en chunks para mejor búsqueda
            5. **Almacenamiento**: Se guarda en base de datos local
            6. **Indexación**: Se crean embeddings para búsqueda semántica

            **Formatos soportados:**
            - PDF (máximo 50MB)
            - DOCX (máximo 50MB)
            - TXT (máximo 50MB) - Nuevo formato agregado
            """)
        
        uploaded_file = st.file_uploader(
            "Selecciona un archivo",
            type=['pdf', 'docx', 'txt'],
            help="Solo se aceptan archivos PDF, DOCX y TXT de hasta 50MB",
            key=f"file_uploader_{st.session_state.get('file_uploader_key', 0)}"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ Archivo seleccionado: **{uploaded_file.name}**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Tamaño", f"{uploaded_file.size / 1024:.1f} KB")
            with col2:
                st.metric("Tipo", uploaded_file.type.split('/')[-1].upper())
            with col3:
                st.metric("Nombre", uploaded_file.name)
            with col4:
                st.metric("Estado", "Pendiente")
            
            with st.spinner("🔍 Analizando archivo..."):
                analysis_result = process_uploaded_file(uploaded_file, st.session_state.vectorstore)
            
            if analysis_result["success"]:
                st.success("✅ Análisis completado")
                if analysis_result.get("is_duplicate", False):
                    st.warning("⚠️ **Archivo duplicado detectado**")
                    st.info(f"**Razón:** {analysis_result.get('duplicate_reason', 'Archivo duplicado')}")
                    if analysis_result.get('existing_file'):
                        st.info(f"**Archivo existente:** {analysis_result['existing_file']}")
                    
                    if st.button("🔄 Subir otro archivo", type="primary", use_container_width=True):
                        clear_upload_interface()
                        st.rerun()
                    return
                
                if "file_stats" in analysis_result:
                    stats = analysis_result["file_stats"]
                    st.markdown("### 📊 Estadísticas del archivo")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Líneas", stats["lines"])
                    with col2:
                        st.metric("Palabras", stats["words"])
                    with col3:
                        st.metric("Caracteres", stats["characters"])
                
                if analysis_result["auto_comment"]:
                    st.markdown("### 📝 Análisis automático")
                    st.markdown(analysis_result["auto_comment"])
                
                st.markdown("### 🎯 Acciones disponibles")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Guardar archivo", type="primary", use_container_width=True):
                        with st.spinner("Guardando archivo..."):
                            save_result = save_uploaded_file(uploaded_file, st.session_state.vectorstore)
                            if save_result["success"]:
                                with st.spinner("🔄 Actualizando buscador..."):
                                    qa_chain = get_or_create_qa_chain(st.session_state.vectorstore, llm_model)
                                show_modal_notification(
                                    "Archivo Guardado Exitosamente",
                                    f"El archivo '{save_result['filename']}' se ha guardado y procesado correctamente.\n\n"
                                    f"• Documentos agregados: {save_result['documents_added']}\n"
                                    f"• Tamaño: {save_result['file_size'] / 1024:.1f} KB\n"
                                    f"• Buscador actualizado automáticamente"
                                )
                                clear_upload_interface()
                                st.session_state.main_tab = "Chat"
                                st.balloons()
                                st.rerun()
                            else:
                                show_modal_notification(
                                    "Error al Guardar",
                                    f"No se pudo guardar el archivo: {save_result['error']}",
                                    success=False
                                )
                with col2:
                    if st.button("🔄 Subir otro archivo", use_container_width=True):
                        clear_upload_interface()
                        st.rerun()
            else:
                show_modal_notification(
                    "Error en el Análisis",
                    f"No se pudo analizar el archivo: {analysis_result['error']}",
                    success=False
                )
                if "corrupto" in analysis_result['error'].lower():
                    st.info("💡 **Sugerencia:** Verifique que el archivo no esté dañado y trate de abrirlo en su aplicación original.")
                elif "formato" in analysis_result['error'].lower():
                    st.info("💡 **Sugerencia:** Asegúrese de que el archivo sea un PDF o DOCX válido.")
                elif "grande" in analysis_result['error'].lower():
                    st.info("💡 **Sugerencia:** Comprima el archivo o divida el contenido en archivos más pequeños.")
                if st.button("🔄 Intentar de nuevo"):
                    clear_upload_interface()
                    st.rerun()

    elif st.session_state.main_tab == "Limpiar base de datos":
        st.markdown("""
        <div class="content-container fade-in-up">
            <h3>🗑️ Limpiar Base de Datos</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Botón para volver al inicio
        if st.button("🏠 Volver al Inicio", use_container_width=True, key="db_home"):
            st.session_state.main_tab = "Principal"
            st.rerun()

        st.divider()
        
        st.markdown("""
        <div class="error-notification fade-in-up">
            <h4>⚠️ Advertencia Importante</h4>
            <p>Esta acción eliminará todos los datos de la base de datos y la regenerará desde cero. Esta operación no se puede deshacer.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("¿Seguro? Limpiar y regenerar base de datos", type="primary", use_container_width=True):
            clear_and_regenerate_database()
            st.session_state.main_tab = "Principal"
            st.rerun()

# =============================================================================
# FUNCIONES DE LOADER PERSONALIZADO
# =============================================================================

def show_custom_loader(message="Cargando..."):
    """Muestra el loader personalizado del Banco de Corrientes"""
    st.markdown(f"""
    <script>
    if (typeof window.showCorrientesLoader === 'function') {{
        window.showCorrientesLoader("{message}");
    }}
    </script>
    """, unsafe_allow_html=True)

def hide_custom_loader():
    """Oculta el loader personalizado"""
    st.markdown("""
    <script>
    if (typeof window.hideCorrientesLoader === 'function') {
        window.hideCorrientesLoader();
    }
    </script>
    """, unsafe_allow_html=True)

# =============================================================================
# FUNCIONES DE GESTIÓN OFFLINE
# =============================================================================

def check_offline_models():
    """Verifica que los modelos offline estén disponibles sin conexión a internet"""
    try:
        # Configurar variables de entorno para modo offline
        import os
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        # Verificar modelo de embeddings desde caché local
        cache_path = Path("./embeddings_cache")
        if cache_path.exists():
            try:
                from sentence_transformers import SentenceTransformer
                # Usar la ruta completa del caché de Hugging Face
                model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./embeddings_cache')
                
                # Probar el modelo con un texto simple
                test_text = "Este es un texto de prueba para verificar el modelo offline."
                embedding = model.encode(test_text)
                
                if embedding and len(embedding) > 0:
                    print(f"✅ Modelo de embeddings '{APP_CONFIG['embedding_model']}' disponible (offline)")
                    return True
                else:
                    print(f"❌ Modelo de embeddings '{APP_CONFIG['embedding_model']}' no responde correctamente")
                    return False
            except Exception as e:
                print(f"❌ Error verificando modelo de embeddings offline: {e}")
                return False
        else:
            print(f"❌ Caché local de embeddings no encontrado en ./embeddings_cache")
            return False
            
    except Exception as e:
        print(f"❌ Error verificando modelo de embeddings: {e}")
        return False

def download_offline_models():
    """Descarga los modelos necesarios para funcionamiento offline"""
    try:
        print("🔄 Verificando modelos offline...")
        
        # Solo verificar si los modelos están disponibles localmente
        cache_path = Path("./embeddings_cache")
        if cache_path.exists():
            print(f"✅ Modelo '{APP_CONFIG['embedding_model']}' ya está disponible offline")
            return True
        else:
            print(f"❌ Modelo '{APP_CONFIG['embedding_model']}' no está disponible offline")
            print("💡 Ejecuta 'python setup_offline.py' con conexión a internet para descargar modelos")
            return False
            
    except Exception as e:
        print(f"❌ Error verificando modelos offline: {e}")
        return False

def get_offline_llm():
    """Obtiene un LLM offline usando modelos locales"""
    # Priorizar modelos locales que funcionan offline
    models_to_try = [
        "mistral",  # Modelo principal optimizado
        "mistral:7b",  # Versión específica de Mistral
        "mistral:latest",  # Última versión de Mistral
        "llama2:7b",  # Fallback a Llama2
        "llama2",  # Fallback genérico
        "codellama:7b"  # Fallback para código
    ]
    
    for model in models_to_try:
        try:
            print(f"🔄 Intentando conectar con modelo offline: {model}")
            
            # Mostrar loader personalizado
            show_custom_loader(f"🏦 Conectando con {model} (offline)...")
            
            # Configuración optimizada para funcionamiento offline
            llm_config = {
                "model": model,
                "temperature": 0.1,  # Baja temperatura para respuestas más consistentes
                "top_p": 0.9,  # Control de diversidad
                "top_k": 40,  # Control de tokens
                "repeat_penalty": 1.1,  # Evitar repeticiones
                "num_ctx": 4096,  # Contexto ampliado
            }
            
            llm = Ollama(**llm_config)
            
            # Actualizar mensaje del loader
            show_custom_loader(f"🏦 Probando {model} (offline)...")
            
            # Probar una consulta simple para verificar que funciona offline
            test_response = llm.invoke("Hola, ¿estás funcionando en modo offline?")
            
            if test_response and len(test_response.strip()) > 0:
                print(f"✅ Modelo offline {model} conectado exitosamente")
                print(f"📊 Configuración offline: temp={llm_config['temperature']}, ctx={llm_config['num_ctx']}")
                
                # Ocultar loader al conectar exitosamente
                hide_custom_loader()
                return llm
            else:
                print(f"⚠️ Modelo offline {model} respondió vacío")
                continue
                
        except Exception as e:
            print(f"❌ Error con modelo offline {model}: {str(e)[:100]}...")
            continue
    
    # Si ningún modelo funciona, mostrar información útil
    print("⚠️ No se pudo conectar con ningún modelo offline")
    print("💡 Sugerencias para funcionamiento offline:")
    print("   1. Verifica que Ollama esté ejecutándose: ollama serve")
    print("   2. Descarga un modelo: ollama pull mistral")
    print("   3. Verifica la conexión: ollama list")
    print("   4. Asegúrate de tener suficiente espacio en disco")
    
    # Ocultar loader al fallar
    hide_custom_loader()
    return None

if __name__ == "__main__":
    main()