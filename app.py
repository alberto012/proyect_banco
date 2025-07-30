import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
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
from langchain.schema import BaseRetriever
from typing import List, Any
from pydantic import Field
import hashlib
import difflib

DOCUMENTS_DIR = Path("./documents")
CHROMA_DB_DIR = Path("./chroma_db")

def normalize_text(text):
    """Normaliza el texto para mejorar la búsqueda"""
    # Convertir a minúsculas
    text = text.lower()
    # Normalizar caracteres Unicode
    text = unicodedata.normalize('NFD', text)
    # Remover acentos
    text = ''.join(c for c in text if not unicodedata.combining(c))
    # Remover caracteres especiales pero mantener espacios
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalizar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def enhance_document_content(doc):
    """Mejora el contenido del documento para búsqueda"""
    original_content = doc.page_content
    normalized_content = normalize_text(original_content)
    
    # Agregar el contenido normalizado al metadata para búsqueda adicional
    doc.metadata['normalized_content'] = normalized_content
    doc.metadata['original_filename'] = doc.metadata.get('source', '')
    
    # También agregar palabras clave extraídas del nombre del archivo
    filename = doc.metadata.get('source', '')
    if filename:
        filename_keywords = normalize_text(Path(filename).stem)
        doc.metadata['filename_keywords'] = filename_keywords
    
    return doc

def enhanced_search(vectorstore, query, k=10):
    """Búsqueda mejorada que combina embeddings con búsqueda de texto"""
    normalized_query = normalize_text(query)
    
    # Generar variaciones de la consulta para búsqueda más flexible
    query_variations = [normalized_query]
    
    # Agregar variaciones comunes y términos relacionados
    if 'complementaria' in normalized_query:
        query_variations.extend([
            'complementaria', 'complementarias', 'complementario', 'complementarios',
            'complementar', 'complementacion', 'complementado', 'complementada'
        ])
    if 'agencia' in normalized_query:
        query_variations.extend(['agencia', 'agencias', 'agencial'])
    if 'evidencia' in normalized_query:
        query_variations.extend(['evidencia', 'evidencias', 'evidenciar', 'evidenciado'])
    
    # Búsqueda semántica con embeddings
    semantic_results = vectorstore.similarity_search(query, k=k)
    
    # Búsqueda adicional por texto normalizado
    text_results = []
    try:
        # Buscar en contenido normalizado
        all_docs = vectorstore.get()
        for i, doc in enumerate(all_docs['documents']):
            metadata = all_docs['metadatas'][i]
            normalized_content = metadata.get('normalized_content', '')
            filename_keywords = metadata.get('filename_keywords', '')
            
            # Buscar coincidencias con todas las variaciones
            found_match = False
            match_score = 0
            
            for variation in query_variations:
                # Buscar en contenido normalizado
                if variation in normalized_content:
                    found_match = True
                    # Calcular score basado en frecuencia y posición
                    count = normalized_content.count(variation)
                    match_score += count * 10
                
                # Buscar en palabras clave del nombre del archivo
                if variation in filename_keywords:
                    found_match = True
                    match_score += 50  # Mayor peso para coincidencias en nombres de archivo
            
            # Búsqueda adicional: buscar términos que contengan la palabra buscada
            if not found_match:
                # Buscar palabras que contengan el término buscado
                words_in_content = normalized_content.split()
                for word in words_in_content:
                    if any(variation in word for variation in query_variations):
                        found_match = True
                        match_score += 5
                        break
            
            if found_match:
                # Crear documento para el resultado
                result_doc = Document(
                    page_content=doc,
                    metadata=metadata
                )
                # Agregar score al metadata para ordenamiento
                result_doc.metadata['match_score'] = match_score
                text_results.append(result_doc)
                
    except Exception as e:
        print(f"Error en búsqueda de texto: {e}")
    
    # Combinar y deduplicar resultados
    combined_results = semantic_results.copy()
    
    # Agregar resultados de texto que no estén ya en los semánticos
    for text_result in text_results:
        # Verificar si ya existe en los resultados semánticos
        exists = False
        for semantic_result in semantic_results:
            if (text_result.page_content == semantic_result.page_content and 
                text_result.metadata.get('source') == semantic_result.metadata.get('source')):
                exists = True
                break
        
        if not exists:
            combined_results.append(text_result)
    
    # Ordenar por score de coincidencia (si está disponible)
    combined_results.sort(key=lambda x: x.metadata.get('match_score', 0), reverse=True)
    
    # Limitar a k resultados
    return combined_results[:k]

class EnhancedRetriever(BaseRetriever):
    """Retriever personalizado que usa búsqueda mejorada"""
    
    vectorstore: Any = Field(description="Vectorstore to search in")
    k: int = Field(default=10, description="Number of documents to retrieve")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Obtiene documentos relevantes usando búsqueda mejorada"""
        return enhanced_search(self.vectorstore, query, self.k)
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Versión asíncrona de _get_relevant_documents"""
        return self._get_relevant_documents(query)

def setup_page_config():
    st.set_page_config(
        page_title="RAG Local",
        layout="centered",
        initial_sidebar_state="auto"
    )
    st.markdown("""
    <style>
    :root {
        --primary: #001CE3;
        --primary-light: #E6F0FF;
        --primary-dark: #007c91;
        --accent: #0012B5;
        --accent-light: #ede7f6;
        --success: #43a047;
        --warning: #fbc02d;
        --danger: #e53935;
        --background: #f6fafd;
        --sidebar-bg: #f0f4f8;
        --card-bg: #ffffff;
        --text-main: #474242;
        --text-secondary: #4f5b62;
        --border: #e0e7ef;
    }
    body, .main, .block-container {
        background: var(--background) !important;
        color: var(--text-main) !important;
    }
    .stApp {
        background: var(--background) !important;
    }
    .stButton>button, .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 7px !important;
        border: 1.5px solid var(--border) !important;
        background: var(--card-bg) !important;
        color: var(--text-main) !important;
        transition: border 0.2s, box-shadow 0.2s;
    }
    .stButton>button {
        background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%) !important;
        color: #fff !important;
        border: none !important;
        margin-bottom: 0.5rem !important;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(10,182,193,0.07);
        transition: background 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, var(--primary-dark) 0%, var(--accent) 100%) !important;
        box-shadow: 0 4px 16px rgba(59,182,193,0.13);
    }
    .stSidebar {
        background: var(--sidebar-bg) !important;
        border-right: 1.5px solid var(--border) !important;
    }
    .stChatMessage {
        background: var(--card-bg) !important;
        border-radius: 10px !important;
        margin-bottom: 0.7rem !important;
        padding: 0.85rem !important;
        box-shadow: 0 2px 8px rgba(59,182,193,0.06);
        border-left: 4px solid var(--primary-light);
    }
    .stChatMessage.user {
        background: var(--primary-light) !important;
        border-left: 4px solid var(--primary) !important;
        text-align: right;
    }
    .stChatMessage.assistant {
        background: var(--accent-light) !important;
        border-left: 4px solid var(--accent) !important;
        text-align: left;
    }
    .stMarkdown, .stExpander, .stSubheader, .stCaption, .stTitle {
        color: var(--text-main) !important;
    }
    .source-card {
        background: var(--primary-light);
        border-left: 4px solid var(--primary);
        padding: 10px 14px;
        margin-bottom: 7px;
        border-radius: 6px;
        font-size: 0.97em;
        box-shadow: 0 1px 4px rgba(59,182,193,0.07);
    }
    .source-card strong {
        color: var(--primary-dark);
    }
    .source-card blockquote {
        color: var(--text-secondary) !important;
        margin: 0.5em 0 0 0;
    }
    details {
        background: var(--accent-light);
        border-radius: 7px;
        padding: 10px;
        margin-top: 12px;
        border: 1.5px solid var(--border);
    }
    summary {
        cursor: pointer;
        font-weight: 600;
        color: var(--accent);
        font-size: 1.05em;
    }
    .st-bb, .st-cq, .st-cp, .st-cq, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz, .st-da, .st-db, .st-dc, .st-dd, .st-de, .st-df, .st-dg, .st-dh, .st-di, .st-dj, .st-dk, .st-dl, .st-dm, .st-dn, .st-do, .st-dp, .st-dq, .st-dr, .st-ds, .st-dt, .st-du, .st-dv, .st-dw, .st-dx, .st-dy, .st-dz, .st-e0, .st-e1, .st-e2, .st-e3, .st-e4, .st-e5, .st-e6, .st-e7, .st-e8, .st-e9, .st-ea, .st-eb, .st-ec, .st-ed, .st-ee, .st-ef, .st-eg, .st-eh, .st-ei, .st-ej, .st-ek, .st-el, .st-em, .st-en, .st-eo, .st-ep, .st-eq, .st-er, .st-es, .st-et, .st-eu, .st-ev, .st-ew, .st-ex, .st-ey, .st-ez, .st-f0, .st-f1, .st-f2, .st-f3, .st-f4, .st-f5, .st-f6, .st-f7, .st-f8, .st-f9, .st-fa, .st-fb, .st-fc, .st-fd, .st-fe, .st-ff, .st-fg, .st-fh, .st-fi, .st-fj, .st-fk, .st-fl, .st-fm, .st-fn, .st-fo, .st-fp, .st-fq, .st-fr, .st-fs, .st-ft, .st-fu, .st-fv, .st-fw, .st-fx, .st-fy, .st-fz, .st-ga, .st-gb, .st-gc, .st-gd, .st-ge, .st-gf, .st-gg, .st-gh, .st-gi, .st-gj, .st-gk, .st-gl, .st-gm, .st-gn, .st-go, .st-gp, .st-gq, .st-gr, .st-gs, .st-gt, .st-gu, .st-gv, .st-gw, .st-gx, .st-gy, .st-gz, .st-ha, .st-hb, .st-hc, .st-hd, .st-he, .st-hf, .st-hg, .st-hh, .st-hi, .st-hj, .st-hk, .st-hl, .st-hm, .st-hn, .st-ho, .st-hp, .st-hq, .st-hr, .st-hs, .st-ht, .st-hu, .st-hv, .st-hw, .st-hx, .st-hy, .st-hz, .st-ia, .st-ib, .st-ic, .st-id, .st-ie, .st-if, .st-ig, .st-ih, .st-ii, .st-ij, .st-ik, .st-il, .st-im, .st-in, .st-io, .st-ip, .st-iq, .st-ir, .st-is, .st-it, .st-iu, .st-iv, .st-iw, .st-ix, .st-iy, .st-iz, .st-ja, .st-jb, .st-jc, .st-jd, .st-je, .st-jf, .st-jg, .st-jh, .st-ji, .st-jj, .st-jk, .st-jl, .st-jm, .st-jn, .st-jo, .st-jp, .st-jq, .st-jr, .st-js, .st-jt, .st-ju, .st-jv, .st-jw, .st-jx, .st-jy, .st-jz, .st-ka, .st-kb, .st-kc, .st-kd, .st-ke, .st-kf, .st-kg, .st-kh, .st-ki, .st-kj, .st-kk, .st-kl, .st-km, .st-kn, .st-ko, .st-kp, .st-kq, .st-kr, .st-ks, .st-kt, .st-ku, .st-kv, .st-kw, .st-kx, .st-ky, .st-kz, .st-la, .st-lb, .st-lc, .st-ld, .st-le, .st-lf, .st-lg, .st-lh, .st-li, .st-lj, .st-lk, .st-ll, .st-lm, .st-ln, .st-lo, .st-lp, .st-lq, .st-lr, .st-ls, .st-lt, .st-lu, .st-lv, .st-lw, .st-lx, .st-ly, .st-lz, .st-ma, .st-mb, .st-mc, .st-md, .st-me, .st-mf, .st-mg, .st-mh, .st-mi, .st-mj, .st-mk, .st-ml, .st-mm, .st-mn, .st-mo, .st-mp, .st-mq, .st-mr, .st-ms, .st-mt, .st-mu, .st-mv, .st-mw, .st-mx, .st-my, .st-mz, .st-na, .st-nb, .st-nc, .st-nd, .st-ne, .st-nf, .st-ng, .st-nh, .st-ni, .st-nj, .st-nk, .st-nl, .st-nm, .st-nn, .st-no, .st-np, .st-nq, .st-nr, .st-ns, .st-nt, .st-nu, .st-nv, .st-nw, .st-nx, .st-ny, .st-nz, .st-oa, .st-ob, .st-oc, .st-od, .st-oe, .st-of, .st-og, .st-oh, .st-oi, .st-oj, .st-ok, .st-ol, .st-om, .st-on, .st-oo, .st-op, .st-oq, .st-or, .st-os, .st-ot, .st-ou, .st-ov, .st-ow, .st-ox, .st-oy, .st-oz, .st-pa, .st-pb, .st-pc, .st-pd, .st-pe, .st-pf, .st-pg, .st-ph, .st-pi, .st-pj, .st-pk, .st-pl, .st-pm, .st-pn, .st-po, .st-pp, .st-pq, .st-pr, .st-ps, .st-pt, .st-pu, .st-pv, .st-pw, .st-px, .st-py, .st-pz, .st-qa, .st-qb, .st-qc, .st-qd, .st-qe, .st-qf, .st-qg, .st-qh, .st-qi, .st-qj, .st-qk, .st-ql, .st-qm, .st-qn, .st-qq, .st-qr, .st-qs, .st-qt, .st-qu, .st-qv, .st-qw, .st-qx, .st-qy, .st-qz, .st-ra, .st-rb, .st-rc, .st-rd, .st-re, .st-rf, .st-rg, .st-rh, .st-ri, .st-rj, .st-rk, .st-rl, .st-rm, .st-rn, .st-ro, .st-rp, .st-rq, .st-rr, .st-rs, .st-rt, .st-ru, .st-rv, .st-rw, .st-rx, .st-ry, .st-rz, .st-sa, .st-sb, .st-sc, .st-sd, .st-se, .st-sf, .st-sg, .st-sh, .st-si, .st-sj, .st-sk, .st-sl, .st-sm, .st-sn, .st-so, .st-sp, .st-sq, .st-sr, .st-ss, .st-st, .st-su, .st-sv, .st-sw, .st-sx, .st-sy, .st-sz, .st-ta, .st-tb, .st-tc, .st-td, .st-te, .st-tf, .st-tg, .st-th, .st-ti, .st-tj, .st-tk, .st-tl, .st-tm, .st-tn, .st-to, .st-tp, .st-tq, .st-tr, .st-ts, .st-tt, .st-tu, .st-tv, .st-tw, .st-tx, .st-ty, .st-tz, .st-ua, .st-ub, .st-uc, .st-ud, .st-ue, .st-uf, .st-ug, .st-uh, .st-ui, .st-uj, .st-uk, .st-ul, .st-um, .st-un, .st-uo, .st-up, .st-uq, .st-ur, .st-us, .st-ut, .st-uu, .st-uv, .st-uw, .st-ux, .st-uy, .st-uz, .st-va, .st-vb, .st-vc, .st-vd, .st-ve, .st-vf, .st-vg, .st-vh, .st-vi, .st-vj, .st-vk, .st-vl, .st-vm, .st-vn, .st-vo, .st-vp, .st-vq, .st-vr, .st-vs, .st-vt, .st-vu, .st-vv, .st-vw, .st-vx, .st-vy, .st-vz, .st-wa, .st-wb, .st-wc, .st-wd, .st-we, .st-wf, .st-wg, .st-wh, .st-wi, .st-wj, .st-wk, .st-wl, .st-wm, .st-wn, .st-wo, .st-wp, .st-wq, .st-wr, .st-ws, .st-wt, .st-wu, .st-wv, .st-ww, .st-wx, .st-wy, .st-wz, .st-xa, .st-xb, .st-xc, .st-xd, .st-xe, .st-xf, .st-xg, .st-xh, .st-xi, .st-xj, .st-xk, .st-xl, .st-xm, .st-xn, .st-xo, .st-xp, .st-xq, .st-xr, .st-xs, .st-xt, .st-xu, .st-xv, .st-xw, .st-xx, .st-xy, .st-xz, .st-ya, .st-yb, .st-yc, .st-yd, .st-ye, .st-yf, .st-yg, .st-yh, .st-yi, .st-yj, .st-yk, .st-yl, .st-ym, .st-yn, .st-yo, .st-yp, .st-yq, .st-yr, .st-ys, .st-yt, .st-yu, .st-yv, .st-yw, .st-yx, .st-yy, .st-yz, .st-za, .st-zb, .st-zc, .st-zd, .st-ze, .st-zf, .st-zg, .st-zh, .st-zi, .st-zj, .st-zk, .st-zl, .st-zm, .st-zn, .st-zo, .st-zp, .st-zq, .st-zr, .st-zs, .st-zt, .st-zu, .st-zv, .st-zw, .st-zx, .st-zy, .st-zz {
        color: var(--text-main) !important;
    }
    </style>
    """, unsafe_allow_html=True)

def load_document_async(filepath):
    try:
        if filepath.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(filepath))
        elif filepath.suffix.lower() == ".docx":
            loader = Docx2txtLoader(str(filepath))
        else:
            return []
        return loader.load()
    except Exception:
        return []

async def process_documents_in_folder_async(documents_path: Path) -> list[Document]:
    loop = asyncio.get_event_loop()
    all_documents = []
    if not documents_path.exists():
        st.info(f"Coloca tus documentos en la carpeta '{documents_path.name}'.")
        return []
    filepaths = [f for f in documents_path.iterdir() if f.suffix.lower() in [".pdf", ".docx"]]
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        tasks = [loop.run_in_executor(executor, load_document_async, filepath) for filepath in filepaths]
        results = await asyncio.gather(*tasks)
    for docs in results:
        all_documents.extend(docs)
    if not all_documents:
        return []
    try:
        # Mejorar el chunking para preservar palabras clave
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Chunks más grandes para mantener contexto
            chunk_overlap=200,  # Más overlap para no perder palabras clave
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Separadores más inteligentes
        )
        chunks = text_splitter.split_documents(all_documents)
        
        # Aplicar mejoras a cada chunk
        enhanced_chunks = []
        for chunk in chunks:
            enhanced_chunk = enhance_document_content(chunk)
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    except Exception:
        return []

@st.cache_resource(show_spinner="Cargando embeddings...")
def get_embeddings_model():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Cargando modelo...") 
def get_ollama_llm():
    return Ollama(model="llama3", base_url="http://localhost:11434")

def initialize_vectorstore_async(documents_dir: Path, chroma_db_dir: Path):
    embeddings_model = get_embeddings_model()
    if chroma_db_dir.exists() and any(chroma_db_dir.iterdir()):
        return Chroma(
            embedding_function=embeddings_model,
            persist_directory=str(chroma_db_dir)
        )
    # Ejecuta la carga asíncrona de documentos
    docs_to_add = asyncio.run(process_documents_in_folder_async(documents_dir))
    if docs_to_add:
        vectorstore = Chroma.from_documents(
            documents=docs_to_add,
            embedding=embeddings_model,
            persist_directory=str(chroma_db_dir)
        )
        vectorstore.persist()
        return vectorstore
    vectorstore = Chroma(
        embedding_function=embeddings_model,
        persist_directory=str(chroma_db_dir)
    )
    vectorstore.persist()
    return vectorstore

def add_comment_to_db(comment_text: str, vectorstore):
    if not comment_text or not vectorstore:
        return None
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        comment_doc = Document(
            page_content=comment_text,
            metadata={
                "source": "Comentario del usuario",
                "creation_date": current_time
            }
        )
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        comment_chunks = text_splitter.split_documents([comment_doc])
        vectorstore.add_documents(comment_chunks)
        vectorstore.persist()
        return current_time
    except Exception:
        return None

def display_saved_comments(vectorstore):
    st.subheader("Comentarios guardados")
    if not vectorstore:
        st.info("Base de datos no disponible.")
        return
    try:
        # Usar búsqueda mejorada para comentarios
        all_comments_docs = enhanced_search(
            vectorstore,
            "comentario del usuario",
            k=1000
        )
        comments_filtered = [doc for doc in all_comments_docs if doc.metadata.get('source') == "Comentario del usuario"]
        if comments_filtered:
            sorted_comments = sorted(
                comments_filtered,
                key=lambda x: datetime.strptime(x.metadata.get('creation_date', '1970-01-01 00:00:00'), "%Y-%m-%d %H:%M:%S"),
                reverse=True
            )
            for doc in sorted_comments:
                st.markdown(f"<span style='color:var(--primary);font-weight:600'>{doc.metadata.get('creation_date', 'N/A')}</span>", unsafe_allow_html=True)
                st.markdown(f"<div style='background:var(--primary-light);padding:0.7em 1em;border-radius:6px;margin-bottom:0.5em;color:var(--text-secondary);'>{doc.page_content}</div>", unsafe_allow_html=True)
        else:
            st.info("No hay comentarios guardados.")
    except Exception:
        st.info("No se pudieron cargar los comentarios.")

def get_loaded_document_names(documents_path: Path) -> list[str]:
    if not documents_path.exists():
        return []
    # Obtener nombres de archivos PDF y DOCX, insensible a mayúsculas/minúsculas
    doc_names = [f.name for f in documents_path.iterdir() if f.suffix.lower() in [".pdf", ".docx"]]
    # Ordenar ignorando mayúsculas/minúsculas
    return sorted(doc_names, key=lambda x: x.lower())

def set_chat_mode():
    st.session_state["show_chat_interface"] = True
    st.session_state["show_comment_input_area"] = False
    st.session_state["show_saved_comments_section"] = False
    st.session_state["show_post_save_return_option"] = False

def set_add_comment_mode():
    st.session_state["show_comment_input_area"] = True
    st.session_state["comment_text_value"] = ""
    st.session_state["show_post_save_return_option"] = False
    st.session_state["show_saved_comments_section"] = False
    st.session_state["show_chat_interface"] = False

def set_view_comments_mode():
    st.session_state["show_saved_comments_section"] = True
    st.session_state["show_comment_input_area"] = False
    st.session_state["show_post_save_return_option"] = False
    st.session_state["show_chat_interface"] = False

def set_upload_file_mode():
    st.session_state["show_upload_file_area"] = True
    st.session_state["show_comment_input_area"] = False
    st.session_state["show_saved_comments_section"] = False
    st.session_state["show_post_save_return_option"] = False
    st.session_state["show_chat_interface"] = False

def clear_chat_history():
    st.session_state["messages"] = []
    set_chat_mode()

def clear_and_regenerate_database():
    """Limpia la base de datos y la regenera con las mejoras"""
    try:
        import shutil
        if CHROMA_DB_DIR.exists():
            shutil.rmtree(CHROMA_DB_DIR)
        
        # Limpiar cache de Streamlit
        st.cache_resource.clear()
        
        # Limpiar el QA chain para que se regenere
        if "qa_chain" in st.session_state:
            del st.session_state["qa_chain"]
        
        st.success("Base de datos limpiada. Se regenerará automáticamente con las mejoras.")
        st.rerun()  # Recargar la página para aplicar cambios
        return True
    except Exception as e:
        st.error(f"Error al limpiar la base de datos: {e}")
        return False

def get_document_hash(content: str) -> str:
    """Calcula el hash MD5 del contenido del documento"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def compare_documents(new_content: str, existing_content: str) -> dict:
    """Compara dos documentos y retorna las diferencias"""
    new_lines = new_content.split('\n')
    existing_lines = existing_content.split('\n')
    
    # Usar difflib para encontrar diferencias
    differ = difflib.Differ()
    diff = list(differ.compare(existing_lines, new_lines))
    
    # Analizar diferencias
    additions = []
    deletions = []
    changes = []
    
    for line in diff:
        if line.startswith('+ '):
            additions.append(line[2:])
        elif line.startswith('- '):
            deletions.append(line[2:])
        elif line.startswith('? '):
            changes.append(line[2:])
    
    return {
        'additions': additions,
        'deletions': deletions,
        'changes': changes,
        'has_changes': len(additions) > 0 or len(deletions) > 0,
        'total_additions': len(additions),
        'total_deletions': len(deletions)
    }

def find_similar_documents(filename: str, vectorstore) -> List[dict]:
    """Busca documentos similares basándose en el nombre del archivo"""
    if not vectorstore:
        return []
    
    try:
        # Buscar por nombre de archivo
        filename_base = Path(filename).stem.lower()
        similar_docs = enhanced_search(vectorstore, filename_base, k=5)
        
        results = []
        for doc in similar_docs:
            source = doc.metadata.get('source', '')
            if source and source != "Comentario del usuario":
                source_filename = Path(source).name
                source_base = Path(source_filename).stem.lower()
                
                # Calcular similitud de nombres
                similarity = difflib.SequenceMatcher(None, filename_base, source_base).ratio()
                
                if similarity > 0.3:  # Umbral de similitud
                    results.append({
                        'filename': source_filename,
                        'full_path': source,
                        'content': doc.page_content,
                        'similarity': similarity,
                        'metadata': doc.metadata
                    })
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)
    except Exception as e:
        print(f"Error buscando documentos similares: {e}")
        return []

def generate_comparison_comment(filename: str, comparison_result: dict, similar_docs: List[dict]) -> str:
    """Genera un comentario automático sobre los cambios detectados"""
    comment_parts = []
    
    # Información básica
    comment_parts.append(f"📄 **Análisis automático del archivo: {filename}**")
    comment_parts.append(f"📅 Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Resumen de cambios
    if comparison_result['has_changes']:
        comment_parts.append(f"\n🔄 **Cambios detectados:**")
        comment_parts.append(f"• Líneas agregadas: {comparison_result['total_additions']}")
        comment_parts.append(f"• Líneas eliminadas: {comparison_result['total_deletions']}")
        
        # Detalles de cambios
        if comparison_result['additions']:
            comment_parts.append(f"\n➕ **Contenido nuevo:**")
            for i, addition in enumerate(comparison_result['additions'][:5]):  # Mostrar solo las primeras 5
                comment_parts.append(f"  - {addition[:100]}{'...' if len(addition) > 100 else ''}")
            if len(comparison_result['additions']) > 5:
                comment_parts.append(f"  ... y {len(comparison_result['additions']) - 5} líneas más")
        
        if comparison_result['deletions']:
            comment_parts.append(f"\n➖ **Contenido eliminado:**")
            for i, deletion in enumerate(comparison_result['deletions'][:5]):  # Mostrar solo las primeras 5
                comment_parts.append(f"  - {deletion[:100]}{'...' if len(deletion) > 100 else ''}")
            if len(comparison_result['deletions']) > 5:
                comment_parts.append(f"  ... y {len(comparison_result['deletions']) - 5} líneas más")
    else:
        comment_parts.append(f"\n✅ **No se detectaron cambios significativos**")
    
    # Documentos similares encontrados
    if similar_docs:
        comment_parts.append(f"\n📚 **Documentos similares encontrados:**")
        for i, doc in enumerate(similar_docs[:3]):  # Mostrar solo los 3 más similares
            similarity_percent = int(doc['similarity'] * 100)
            comment_parts.append(f"• {doc['filename']} (similitud: {similarity_percent}%)")
    
    return "\n".join(comment_parts)

def process_uploaded_file(uploaded_file, vectorstore) -> dict:
    """Procesa un archivo subido y lo compara con documentos existentes"""
    try:
        # Leer el contenido del archivo subido
        if uploaded_file.type == "application/pdf":
            # Para PDFs, necesitamos guardarlo temporalmente
            temp_path = DOCUMENTS_DIR / f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader(str(temp_path))
            new_docs = loader.load()
            
            # Limpiar archivo temporal
            temp_path.unlink()
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Para DOCX, también guardar temporalmente
            temp_path = DOCUMENTS_DIR / f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = Docx2txtLoader(str(temp_path))
            new_docs = loader.load()
            
            # Limpiar archivo temporal
            temp_path.unlink()
        else:
            return {"success": False, "error": "Formato de archivo no soportado"}
        
        if not new_docs:
            return {"success": False, "error": "No se pudo leer el contenido del archivo"}
        
        # Combinar todo el contenido del nuevo documento
        new_content = "\n".join([doc.page_content for doc in new_docs])
        new_hash = get_document_hash(new_content)
        
        # Buscar documentos similares
        similar_docs = find_similar_documents(uploaded_file.name, vectorstore)
        
        # Comparar con documentos similares
        comparison_results = []
        for similar_doc in similar_docs:
            comparison = compare_documents(new_content, similar_doc['content'])
            if comparison['has_changes']:
                comparison['similar_doc'] = similar_doc
                comparison_results.append(comparison)
        
        # Generar comentario automático si hay cambios
        auto_comment = None
        if comparison_results:
            # Usar el documento más similar para la comparación
            best_comparison = max(comparison_results, key=lambda x: x['similar_doc']['similarity'])
            auto_comment = generate_comparison_comment(uploaded_file.name, best_comparison, similar_docs)
        
        return {
            "success": True,
            "filename": uploaded_file.name,
            "content": new_content,
            "hash": new_hash,
            "similar_docs": similar_docs,
            "comparison_results": comparison_results,
            "auto_comment": auto_comment,
            "has_changes": len(comparison_results) > 0
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def save_uploaded_file(uploaded_file, vectorstore) -> bool:
    """Guarda el archivo subido en la carpeta de documentos"""
    try:
        # Guardar el archivo
        file_path = DOCUMENTS_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Procesar y agregar a la base de datos
        docs_to_add = load_document_async(file_path)
        if docs_to_add:
            # Aplicar mejoras a cada documento
            enhanced_docs = []
            for doc in docs_to_add:
                enhanced_doc = enhance_document_content(doc)
                enhanced_docs.append(enhanced_doc)
            
            # Agregar a la base de datos
            vectorstore.add_documents(enhanced_docs)
            vectorstore.persist()
            return True
        
        return False
    except Exception as e:
        print(f"Error guardando archivo: {e}")
        return False

def main():
    setup_page_config()

    session_defaults = {
        "vectorstore": None,
        "messages": [],
        "show_comment_input_area": False,
        "comment_text_value": "",
        "show_saved_comments_section": False,
        "show_post_save_return_option": False,
        "show_chat_interface": True,
        "show_upload_file_area": False
    }
    for k, v in session_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    st.markdown(
        "<h2 style='font-weight:700;margin-bottom:0.2em;color:var(--primary);letter-spacing:-1px;'>RAG Local</h2>",
        unsafe_allow_html=True
    )
    st.caption("Consulta tus documentos Word y PDF de forma offline.")

    # Sidebar minimalista y profesional
    with st.sidebar:
        st.markdown("<div style='font-size:1.2em;font-weight:600;color:var(--primary);margin-bottom:0.5em;'>Navegación</div>", unsafe_allow_html=True)
        st.button("💬 Chat", on_click=set_chat_mode, use_container_width=True)
        st.button("📁 Subir archivo", on_click=set_upload_file_mode, use_container_width=True)
        st.button("📝 Añadir comentario", on_click=set_add_comment_mode, use_container_width=True)
        st.button("📋 Ver comentarios", on_click=set_view_comments_mode, use_container_width=True)
        st.button("🧹 Limpiar chat", on_click=clear_chat_history, use_container_width=True)
        st.button("🧹 Limpiar base de datos", on_click=clear_and_regenerate_database, use_container_width=True)
        st.markdown("<hr style='border:1px solid var(--border);margin:0.7em 0;'/>", unsafe_allow_html=True)
        docs = get_loaded_document_names(DOCUMENTS_DIR)
        st.markdown("<span style='font-weight:600;color:var(--accent);'>Documentos:</span>", unsafe_allow_html=True)
        if docs:
            for doc in docs:
                st.markdown(f"<span style='color:var(--text-secondary);font-size:0.98em;'>• {doc}</span>", unsafe_allow_html=True)
        else:
            st.caption("No hay documentos.")

    # Inicialización de componentes
    try:
        llm_model = get_ollama_llm()
        st.session_state.vectorstore = initialize_vectorstore_async(DOCUMENTS_DIR, CHROMA_DB_DIR)
    except Exception:
        st.error("No se pudo inicializar el modelo o la base de datos.")
        return

    # QA Chain
    if st.session_state.vectorstore and llm_model and "qa_chain" not in st.session_state:
        template = (
            "Busca EXHAUSTIVAMENTE en el contexto cualquier información relacionada con la pregunta. "
            "Si la pregunta menciona una palabra específica, busca esa palabra Y TODAS SUS VARIACIONES Y TÉRMINOS RELACIONADOS. "
            "Por ejemplo, si buscas 'complementaria', también busca 'complementarias', 'complementario', 'complementarios', 'complementar', etc. "
            "Si buscas 'agencia', también busca 'agencias', 'agencial', etc. "
            "Busca en títulos de archivos, contenido y comentarios. "
            "Si encuentras información relacionada (incluso si es solo una mención), responde con ella. "
            "Si encuentras frases como 'Agencia complementaria' cuando buscas 'complementaria', esa información ES RELEVANTE. "
            "Solo responde 'no conozco la respuesta, no tengo informacion disponible para responder la pregunta' si realmente no encuentras NINGUNA mención relacionada.\n\n"
            "Contexto: {context}\n\nPregunta: {question}\nRespuesta:"
        )
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
        # Usar el retriever mejorado
        enhanced_retriever = EnhancedRetriever(
            vectorstore=st.session_state.vectorstore, 
            k=15
        )
        
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm_model,
            chain_type="stuff",
            retriever=enhanced_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

    # --- UI Principal ---
    if st.session_state.show_comment_input_area:
        st.subheader("Nuevo comentario")
        new_comment_text = st.text_area(
            "Escribe tu comentario:",
            key="comment_input_main_area",
            value=st.session_state.comment_text_value,
            height=100
        )
        st.session_state.comment_text_value = new_comment_text

        if st.button("Guardar", type="primary"):
            if new_comment_text:
                timestamp = add_comment_to_db(new_comment_text, st.session_state.vectorstore)
                if timestamp:
                    st.success("Comentario guardado.")
                    st.session_state.messages.append({"role": "assistant", "content": f"Comentario guardado ({timestamp})."})
                    st.session_state.comment_text_value = ""
                    st.session_state.show_post_save_return_option = True
            else:
                st.info("Escribe un comentario antes de guardar.")

        if st.session_state.show_post_save_return_option:
            st.button("Volver al chat", key="return_to_chat_from_comment", on_click=set_chat_mode)

    elif st.session_state.show_saved_comments_section:
        display_saved_comments(st.session_state.vectorstore)

    elif st.session_state.show_upload_file_area:
        st.subheader("📁 Subir archivo")
        st.markdown("Sube un archivo PDF o DOCX para agregarlo a tu base de documentos.")
        
        uploaded_file = st.file_uploader(
            "Selecciona un archivo",
            type=['pdf', 'docx'],
            help="Solo se aceptan archivos PDF y DOCX"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ Archivo seleccionado: **{uploaded_file.name}**")
            
            # Mostrar información del archivo
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tamaño", f"{uploaded_file.size / 1024:.1f} KB")
            with col2:
                st.metric("Tipo", uploaded_file.type)
            with col3:
                st.metric("Nombre", uploaded_file.name)
            
            # Procesar archivo para análisis
            with st.spinner("Analizando archivo..."):
                analysis_result = process_uploaded_file(uploaded_file, st.session_state.vectorstore)
            
            if analysis_result["success"]:
                st.success("✅ Análisis completado")
                
                # Mostrar documentos similares encontrados
                if analysis_result["similar_docs"]:
                    st.markdown("### 📚 Documentos similares encontrados")
                    for i, doc in enumerate(analysis_result["similar_docs"][:3]):
                        similarity_percent = int(doc['similarity'] * 100)
                        with st.expander(f"📄 {doc['filename']} (Similitud: {similarity_percent}%)"):
                            st.markdown(f"**Ruta:** `{doc['full_path']}`")
                            st.markdown(f"**Similitud:** {similarity_percent}%")
                            st.markdown("**Vista previa del contenido:**")
                            st.text(doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content'])
                
                # Mostrar análisis de cambios si los hay
                if analysis_result["has_changes"]:
                    st.warning("🔄 Se detectaron cambios en comparación con documentos existentes")
                    
                    # Mostrar comentario automático generado
                    if analysis_result["auto_comment"]:
                        st.markdown("### 📝 Comentario automático generado")
                        st.markdown(analysis_result["auto_comment"])
                        
                        # Opción para guardar el comentario automático
                        if st.button("💾 Guardar comentario automático", type="primary"):
                            timestamp = add_comment_to_db(analysis_result["auto_comment"], st.session_state.vectorstore)
                            if timestamp:
                                st.success("✅ Comentario automático guardado")
                            else:
                                st.error("❌ Error al guardar el comentario")
                else:
                    st.info("ℹ️ No se detectaron cambios significativos con documentos existentes")
                
                # Botones de acción
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Guardar archivo", type="primary", use_container_width=True):
                        with st.spinner("Guardando archivo..."):
                            if save_uploaded_file(uploaded_file, st.session_state.vectorstore):
                                st.success("✅ Archivo guardado exitosamente")
                                st.rerun()  # Recargar para mostrar el nuevo documento
                            else:
                                st.error("❌ Error al guardar el archivo")
                
                with col2:
                    if st.button("🔄 Cancelar", use_container_width=True):
                        st.rerun()
                
            else:
                st.error(f"❌ Error en el análisis: {analysis_result['error']}")
                if st.button("🔄 Intentar de nuevo"):
                    st.rerun()

    # Controlar si el input debe estar deshabilitado
    if "input_disabled" not in st.session_state:
        st.session_state.input_disabled = False

    if st.session_state.show_chat_interface:
        st.subheader("Chat")
        if not st.session_state.messages:
            st.info("Haz una pregunta sobre tus documentos o comentarios.")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)

        # Deshabilitar el input si está procesando
        prompt = st.chat_input("Pregunta...", disabled=st.session_state.input_disabled)

        if prompt:
            # Limpiar historial de mensajes antes de agregar la nueva pregunta
            st.session_state.messages = []
            st.session_state.input_disabled = True
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.spinner("Pensando..."):
                try:
                    if "qa_chain" not in st.session_state or not st.session_state.qa_chain:
                        st.error("El motor de consultas no está listo.")
                        st.session_state.messages.append({"role": "assistant", "content": "El agente no está listo para responder."})
                        st.session_state.input_disabled = False
                        return
                    result = st.session_state.qa_chain.invoke({"query": prompt})
                    response_text = result.get("result", "")
                    source_docs = result.get("source_documents", [])
                    print(f"[QA] Respuesta local: '{response_text}'")
                    # Solo reemplazar si realmente no hay información útil
                    has_useful_info = (
                        response_text.strip() and 
                        response_text.strip().lower() not in ["no lo sé", "no lo se", "no sé", "no se"] and
                        "no conozco la respuesta" not in response_text.lower() and
                        not any(kw in response_text.strip().lower() for kw in ["no tengo información", "no hay información", "no encuentro información"])
                    )
                    
                    if not has_useful_info:
                        response_text = "no conozco la respuesta, no tengo informacion disponible para responder la pregunta"
                        source_docs = []
                        formatted_response = response_text
                        # Eliminar también el mensaje del usuario del historial
                        st.session_state.messages = []
                    else:
                        formatted_response = response_text
                    # Si la respuesta es de tipo 'No entiendo la pregunta...' tampoco mostrar fuentes
                    GENERIC_NO_UNDERSTAND = [
                        "no entiendo la pregunta",
                        "por favor proporciona más contexto",
                        "rephraze la pregunta",
                        "reformula la pregunta"
                    ]
                    if any(kw in response_text.strip().lower() for kw in GENERIC_NO_UNDERSTAND):
                        source_docs = []
                        formatted_response = response_text
                    print(f"[Final] Respuesta mostrada: '{formatted_response}'")
                    # Siempre mostrar fuentes si hay documentos recuperados y la respuesta no es el mensaje genérico
                    if source_docs and response_text.strip() and response_text != "no conozco la respuesta, no tengo informacion disponible para responder la pregunta" and "no conozco la respuesta" not in response_text.lower():
                        # Mostrar todas las fuentes recuperadas
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
                            content_preview = doc.page_content.strip()
                            if len(content_preview) > 300:
                                content_preview = content_preview[:300] + "..."
                            fuentes_utilizadas.append(
                                f"<div class='source-card'><strong>{source_detail}</strong><br><blockquote>{content_preview}</blockquote></div>"
                            )
                        # Siempre mostrar las fuentes si existen
                        full_response_with_sources = (
                            f"{formatted_response}\n\n"
                            f"<details><summary>Fuentes ({len(fuentes_utilizadas)})</summary>\n"
                            f"{''.join(fuentes_utilizadas)}\n</details>"
                        )
                    else:
                        full_response_with_sources = formatted_response
                    with st.chat_message("assistant"):
                        st.markdown(full_response_with_sources, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": full_response_with_sources})
                except Exception:
                    st.session_state.messages.append({"role": "assistant", "content": "Ocurrió un error al procesar la consulta."})
                finally:
                    st.session_state.input_disabled = False
                    st.rerun()  # Forzar actualización de la UI

if __name__ == "__main__":
    main()