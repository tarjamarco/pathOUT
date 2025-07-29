#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
# Enhanced Pathology Chatbot · RAG with Query Expansion and improved retrieval
# ──────────────────────────────────────────────────────────────────────────────
import os, json, pickle, hashlib, textwrap, re, types
import streamlit as st
from dotenv import load_dotenv
import re
from typing import List, Dict, Tuple, Any, Set
import time
import threading
import concurrent.futures # Add for parallel execution
from collections import defaultdict

# vector libs
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

# LLM api
import google.generativeai as genai

# ──────────────────────────────────────────────────────────────────────────────
# 0. Basic setup
# ──────────────────────────────────────────────────────────────────────────────
try:
    import torch
    if not hasattr(torch.classes, "__path__"):
        torch.classes.__path__ = types.SimpleNamespace(_path=[])
except ModuleNotFoundError:
    pass

# Set device for PyTorch operations
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon MPS backend.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA backend.")
else:
    device = torch.device("cpu")
    print("MPS/CUDA backend not available, falling back to CPU.")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Enhanced Config (Modified)
# ──────────────────────────────────────────────────────────────────────────────
ROOT_JSON_DIR   = "json"
EMBED_MODEL_ID  = "neuml/pubmedbert-base-embeddings"
CROSS_ENCODER_ID= "ncbi/MedCPT-Cross-Encoder"
INDEX_PATH      = "faiss.idx"
META_PATH       = "chunk_meta.pkl"
CHUNK_TOKENS    = 100
CHUNK_OVERLAP   = 30
TOP_K_FAST      = 50
TOP_K_FINAL     = 5
MEMORY_MAX_TOK  = 400

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Enhanced DDX Configuration
DDX_CONFIG = {
    'confidence_threshold': 0.6,     # Minimum confidence for DDX detection
    'primary_entity_boost': 1.2,     # Boost for primary entity matches
    'ddx_content_boost': 1.15,       # Boost for explicit DDX content
    'max_candidates': 30,            # Maximum candidates before reranking
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. Utility functions (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def is_comparative_query(query: str) -> Tuple[bool, str, str]:
    """Detects if the query is a comparative (DDx) query and extracts diseases A and B."""
    ddx_patterns = [
        r"(?:differential diagnosis|difference|distinguish|or|vs\.?|versus)\s+(?:between\s+)?([\w\s/-]+?)\s+(?:and|vs\.?|versus)\s+([\w\s/-]+)",
        r"([\w\s/-]+)\s+vs\.?\s+([\w\s/-]+)",
    ]
    for pattern in ddx_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            A, B = match.groups()
            return True, A.strip(), B.strip()
    return False, "", ""

def compute_corpus_signature(root: str) -> str:
    parts = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".json"):
                fp = os.path.join(dp, fn)
                parts.append(f"{fp}:{os.path.getmtime(fp)}")
    return hashlib.md5("|".join(sorted(parts)).encode()).hexdigest()

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def sentence_chunks(txt: str, tokens_per_chunk: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP):
    sents = sent_tokenize(txt)
    cur, cur_len = [], 0
    for s in sents:
        tok = len(s.split())
        if cur_len + tok > tokens_per_chunk and cur:
            yield " ".join(cur)
            cur = cur[-overlap // 15 :]
            cur_len = len(" ".join(cur).split())
        cur.append(s)
        cur_len += tok
    if cur:
        yield " ".join(cur)

def flatten(it):
    for x in it:
        if isinstance(x, list):
            yield from flatten(x)
        else:
            yield x

# ──────────────────────────────────────────────────────────────────────────────
# 3. Enhanced Query Processing (MODIFIED - Added Hypothetical Question)
# ──────────────────────────────────────────────────────────────────────────────

def generate_hypothetical_question(question: str) -> str:
    """Generates a detailed, hypothetical question focused on clinical context to improve retrieval (HyDE)."""
    if st.session_state.get("interrupt_request", False):
        return question

    # This prompt is designed to create a clinically-focused question
    # that a pathologist would ask to correlate with their histological findings.
    hyde_prompt = f"""
A pathologist is examining a case and needs to correlate their findings with the patient's clinical presentation.
Based on the user's query, generate a single, detailed hypothetical question that this pathologist might ask to get the most relevant clinical context.
This question will be used to search a medical database for information to help with the diagnosis.
Focus on **clinical features** such as patient presentation, age, symptoms, lab findings, and relevant medical history, rather than histological details which the pathologist may already know.

Original user query: {question}

Generate only the detailed hypothetical question:"""

    try:
        # Use a fast model for this generation task
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(hyde_prompt)
        hypothetical_question = response.text.strip()
        st.info(f"🧠 Hypothetical Q: *{hypothetical_question}*")
        return hypothetical_question
    except Exception as e:
        st.warning(f"Hypothetical question generation failed: {e}")
        return question # Fallback to original question


def expand_query(question: str) -> List[str]:
    """Generate query variations and synonyms"""
    if st.session_state.get("interrupt_request", False):
        return [question]

    expansion_prompt = f"""
Generate 2-3 alternative ways to ask this pathology question, using different medical terminology and phrasing. Focus on:
- Synonyms for key terms
- Different clinical contexts
- Alternative diagnostic approaches

Original question: {question}

Provide only the alternative questions, one per line:
"""

    try:
        # Use a faster model for expansion if possible
        response = genai.GenerativeModel("gemini-2.5-flash-lite").generate_content(expansion_prompt)
        alternatives = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
        # The original question is added in the main retrieval function
        return alternatives[:3]
    except Exception as e:
        st.warning(f"Query expansion failed: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# 4. Corpus processing (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def load_corpus(root: str):
    corpus = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".json"):
                fp_abs = os.path.join(dp, fn)
                fp_rel = os.path.relpath(fp_abs, root)
                with open(fp_abs) as f:
                    raw = json.load(f)
                    for d in flatten(raw):
                        if isinstance(d, dict):
                            d["_actual_json_file_rel_path"] = fp_rel
                            corpus.append(d)
    return corpus

def corpus_to_chunks(corpus):
    """Enhanced chunk processing with metadata"""
    out = []
    # This part of your code seems to be missing the ChunkMetadataEnhancer class definition
    # For now, we will proceed without it, but you might need to add it back.
    # enhancer = ChunkMetadataEnhancer()

    for sec in corpus:
        actual_json_file_rel_path = sec.get("_actual_json_file_rel_path", "")
        if not actual_json_file_rel_path:
            continue

        original_source_doc_name = sec.get('source_document', 'unknown_file.txt')
        if original_source_doc_name.lower().endswith('.txt'):
            corrected_source_doc_name = original_source_doc_name[:-4] + '.json'
        else:
            corrected_source_doc_name = original_source_doc_name

        display_source_id = f"{sec.get('parent', 'unknown_folder')}/{corrected_source_doc_name}"

        # Use the actual source URL instead of JSON file path
        actual_url = sec.get("actual_source_url", "https://www.pathologyoutlines.com")

        for ch_text in sentence_chunks(sec.get("clean_content", "")):
            chunk = {
                "text": ch_text,
                "topic": sec.get("topic", ""),
                "url": sec.get("url", ""),
                "actual_json_file_path": actual_json_file_rel_path,
                "display_source_id": display_source_id,
            }

            # Add enhanced metadata
            # chunk = enhancer.enhance_chunk_metadata(chunk) # This line would cause an error without the class
            out.append(chunk)

    return out

# ──────────────────────────────────────────────────────────────────────────────
# 5. Enhanced Vector Store (Modified - Models on device)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def load_vector_store(sig: str):
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
            if meta.get("signature") == sig:
                return faiss.read_index(INDEX_PATH), meta["chunks"]

    st.info("🔄 Building enhanced vector store...")
    corpus = load_corpus(ROOT_JSON_DIR)
    chunks = corpus_to_chunks(corpus)

    embedder = SentenceTransformer(EMBED_MODEL_ID)
    embedder.to(device) # Move embedder to device
    vecs = embedder.encode(
        [c["text"] for c in chunks],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump({"signature": sig, "chunks": chunks}, f)
    return index, chunks

# Ensure models are loaded once and moved to device
@st.cache_resource
def get_embedder_single():
    model = SentenceTransformer(EMBED_MODEL_ID)
    model.to(device) # Move model to GPU/MPS
    return model

@st.cache_resource
def get_cross_encoder():
    model = CrossEncoder(CROSS_ENCODER_ID)
    model.to(device) # Move model to GPU/MPS
    return model

embedder_single = get_embedder_single()
cross_encoder = get_cross_encoder()

# ──────────────────────────────────────────────────────────────────────────────
# 6. Enhanced Retrieval with Query Expansion (MODIFIED - Integrated Hypothetical Question)
# ──────────────────────────────────────────────────────────────────────────────

class DDxEntityDetector:
    """
    Detects differential diagnosis (DDx) queries and extracts entities with confidence.
    """
    def extract_entities(self, question: str):
        ddx_patterns = [
            r'(?i)differential\s+diagnosis\s+(?:between|of)\s+([a-zA-Z\s]+)\s+(?:and|vs\.?|versus)\s+([a-zA-Z\s]+)',
            r'(?i)ddx\s+(?:between|of)\s+([a-zA-Z\s]+)\s+(?:and|vs\.?|versus)\s+([a-zA-Z\s]+)',
            r'(?i)difference\s+between\s+([a-zA-Z\s]+)\s+and\s+([a-zA-Z\s]+)'
        ]
        for pattern in ddx_patterns:
            match = re.search(pattern, question)
            if match:
                entity1 = match.group(1).strip()
                entity2 = match.group(2).strip()
                conf = min(1.0, 0.5 + 0.1 * (len(entity1.split()) + len(entity2.split())))
                return (entity1, entity2, conf)
        return None

class DifferentialDiagnosisRetriever:
    """
    Advanced DDx retriever: detects DDx queries, extracts entities, and retrieves relevant chunks.
    """
    def __init__(self, embedder, cross_encoder, index, chunks):
        self.embedder = embedder
        self.cross_encoder = cross_encoder
        self.index = index
        self.chunks = chunks
        self.detector = DDxEntityDetector()

    def retrieve_differential(self, query: str, top_k=DDX_CONFIG['max_candidates']) -> List[Tuple[str, float]]:
        is_ddx, A, B = is_comparative_query(query)
        if not is_ddx:
            return retrieve_basic(query, self.index, self.chunks, top_k=TOP_K_FAST)

        query_vec = self.embedder.encode([query])[0]
        # Your existing DDx logic seems to have a bug where it calls a `retrieve` function
        # and references variables not in scope (`meta`, `embed_fn`).
        # The logic below is a simplified placeholder based on your original intent.
        # You may need to refine this DDx-specific retrieval part.
        st.warning("Note: DDx retrieval logic is using a simplified implementation.")
        # For this example, we'll just fall back to a basic retrieval for the DDx query.
        return retrieve_basic(query, self.index, self.chunks, top_k=top_k)


def retrieve_with_expansion(question: str, index, chunks) -> List[Tuple[float, Dict]]:
    """Enhanced retrieval with HyDE, DDX support, and query expansion"""
    if st.session_state.get("interrupt_request", False):
        return []

    # Initialize the DDx retriever if not already in session state
    if 'ddx_retriever' not in st.session_state:
        st.session_state.ddx_retriever = DifferentialDiagnosisRetriever(
            embedder_single, cross_encoder, index, chunks
        )
    ddx_retriever = st.session_state.ddx_retriever

    # Try differential diagnosis retrieval first
    ddx_result = ddx_retriever.detector.extract_entities(question)
    if ddx_result:
        entity1, entity2, confidence = ddx_result
        if confidence > DDX_CONFIG['confidence_threshold']:
            st.info(f"🔍 Detected differential diagnosis query: **{entity1}** vs **{entity2}**")
            # This calls the method on the instantiated object
            return ddx_retriever.retrieve_differential(question, top_k=TOP_K_FINAL)

    # --- FALLBACK TO HYDE + EXPANSION LOGIC ---
    all_candidates = {}

    # 1. Generate Hypothetical Question (HyDE)
    hypothetical_question = generate_hypothetical_question(question)

    # 2. Expand original query
    expanded_queries = expand_query(question)

    # 3. Create a weighted list of queries for retrieval
    # Higher weight for the hypothetical question
    queries_with_weights = [
        (hypothetical_question, 1.1),
        (question, 1.0)
    ] + [(q, 0.8) for q in expanded_queries]


    for i, (query, weight) in enumerate(queries_with_weights):
        if st.session_state.get("interrupt_request", False):
            break

        # Retrieve candidates for the current query
        exp_candidates = retrieve_basic(query, index, chunks, TOP_K_FAST // (i + 1))

        # Add weighted candidates to the main dictionary, avoiding duplicates
        for score, chunk in exp_candidates:
            chunk_id = id(chunk)
            adjusted_score = score * weight
            if chunk_id not in all_candidates or adjusted_score > all_candidates[chunk_id][0]:
                all_candidates[chunk_id] = (adjusted_score, chunk)

    # Sort all collected candidates by their final adjusted score
    candidate_list = sorted(list(all_candidates.values()), key=lambda x: x[0], reverse=True)

    if st.session_state.get("interrupt_request", False) or not candidate_list:
        return candidate_list[:TOP_K_FINAL]

    # --- CROSS-ENCODER RE-RANKING ---
    # Re-rank the top candidates using the more powerful cross-encoder
    pairs = [[question, chunk["text"]] for _, chunk in candidate_list[:TOP_K_FAST]]
    if pairs:
        cross_scores = cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)
        combined_scores = [(cross_scores[i], chunk) for i, (_, chunk) in enumerate(candidate_list[:len(cross_scores)])]
        combined_scores.sort(key=lambda x: x[0], reverse=True)
        return combined_scores[:TOP_K_FINAL]

    return []


def retrieve_basic(query: str, index, chunks, top_k: int = TOP_K_FAST) -> List[Tuple[float, Dict]]:
    """Basic retrieval without re-ranking"""
    qv = embedder_single.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(qv, min(top_k, len(chunks)))
    return [(float(scores[0][i]), chunks[ids[0][i]]) for i in range(len(ids[0]))]

def retrieve(question: str, index, chunks):
    """Main retrieval function - now uses HyDE, DDx logic, and query expansion"""
    return retrieve_with_expansion(question, index, chunks)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Enhanced Prompt Building - UPDATED FOR SMART CITATION SYSTEM
# ──────────────────────────────────────────────────────────────────────────────

FEW_SHOTS = """
### Example A1
Acute promyelocytic leukaemia (APL) is characterised by abnormal promyelocytes with heavy granulation and the pathognomonic t(15;17) translocation creating PML-RARA fusion ([Source](SOURCE_1)). Auer rods are frequently present in malignant cells, and there is a high risk of DIC due to procoagulant release ([Source](SOURCE_2)).
"""

DEFAULT_SYSTEM_TMPL = textwrap.dedent("""
You are PathOut, an AI assistant from Pathology Outlines, developed by Algoscope, created to support medical professionals—including pathology residents, physicians, and lab technologists—with accurate, concise answers.                                      

Based EXCLUSIVELY on the following text compilation from various documents AND the conversation history if provided, answer the user's latest question. Assume the user is a medical professional who requires specific, technical details. Do not use any prior knowledge outside the context.
Respect the instruction but don't begin your response with "Based on the provided text", begin directly with the response. IF provided text doesn't contain enough information to answer the question, don't begin your response with "The provided text doesn't contain..", you can say something like "I am unable to find the relevant information on [Pathology Outlines](https://www.pathologyoutlines.com)".
If asked about your identity, state that you are PathOut, an AI assistant from Pathology Outlines and developed by Algoscope. Do not mention Google or being a large language model.

**CRITICAL Citation Instructions:** When referencing specific information from the sources, you MUST use this EXACT format:
- Single source: ([Source](SOURCE_N)) where N is the source number (1, 2, 3, etc.)
- Multiple sources: ([Source](SOURCE_1), [Source](SOURCE_2), [Source](SOURCE_3))
- Do NOT write the actual URL - use only SOURCE_N as shown
- The system will automatically convert SOURCE_N to the proper clickable link
- Examples: 
  - Single: ([Source](SOURCE_1))
  - Multiple: ([Source](SOURCE_1), [Source](SOURCE_2))

NEVER write the full URL in your response. Always use the SOURCE_N format.
""").strip()

def build_enhanced_prompt(question, hits, memory=""):
    """Build enhanced prompt with properly numbered sources"""
    system_template = st.session_state.get("system_template", DEFAULT_SYSTEM_TMPL)
    
    if not hits:
        return f"{system_template}\n\n### Current Question\n{question}\n\n### Reference Material\nNo relevant sources found.", {}
    
    ctx = ""
    link_map = {}
    
    # Build context with properly numbered sources (now directly using hits)
    for i, (score, chunk) in enumerate(hits, start=1):
        source_num = i
        source_key = f"SOURCE_{source_num}"
        
        ctx += f"\n### Source {source_num}: {chunk['display_source_id']}\n"
        ctx += f"Content: {chunk['text']}\n"
        if chunk.get('topic'):
            ctx += f"Topic: {chunk['topic']}\n"
        
        link_map[source_key] = {
            "url": chunk["url"],
            "topic": chunk.get("topic", ""),
            "chunk": chunk["text"],
            "display_source_id": chunk["display_source_id"],
            "relevance_score": score
        }
    
    retrieval_info = "Advanced retrieval with Hypothetical Question & Query Expansion"
    
    prompt = f"""{system_template}

{FEW_SHOTS}

### Conversation Context
{memory}

### Current Question
{question}

### Reference Material ({retrieval_info})
{ctx}

Remember: Provide a SINGLE, CONCISE paragraph response with proper SOURCE citations using the format ([Source](SOURCE_N)).
"""
    
    return prompt, link_map

# ──────────────────────────────────────────────────────────────────────────────
# 8. SMART linkification - HANDLES MULTIPLE SOURCES BUT SHOWS SINGLE SOURCE AT END
# ──────────────────────────────────────────────────────────────────────────────
def smart_linkify(text: str, src: dict) -> str:
    """Smart linkification: inline for multiple sources, end citation for single source"""
    
    source_pattern = re.compile(r'SOURCE_(\d+)')
    mentioned_sources = set(source_pattern.findall(text))
    
    if len(mentioned_sources) == 1:
        clean_text = re.sub(r'\(\[Source\]\(SOURCE_\d+\)\)', '', text)
        clean_text = re.sub(r'\[Source\]\(SOURCE_\d+\)', '', clean_text)
        clean_text = re.sub(r'SOURCE_\d+', '', clean_text)
        clean_text = clean_text.strip()
        
        source_num = mentioned_sources.pop()
        source_key = f"SOURCE_{source_num}"
        if source_key in src:
            url = src[source_key].get("url", "")
            if url:
                clean_text += f"\n\n**Source:** [Pathology Outlines]({url})"
        
        return clean_text
    
    multi_source_pattern = re.compile(r'\((\[Source\]\(SOURCE_\d+\)(?:,\s*\[Source\]\(SOURCE_\d+\))*)\)')
    
    def replace_multi_sources(match):
        sources_content = match.group(1)
        individual_sources = re.findall(r'\[Source\]\(SOURCE_(\d+)\)', sources_content)
        
        unique_sources = []
        seen = set()
        for source_num in individual_sources:
            if source_num not in seen:
                unique_sources.append(source_num)
                seen.add(source_num)
        
        links = []
        for source_num in unique_sources:
            source_key = f"SOURCE_{source_num}"
            if source_key in src:
                url = src[source_key].get("url", "")
                if url:
                    links.append(f"[Source]({url})")
                else:
                    links.append(f"[Source](SOURCE_{source_num})")
            else:
                links.append(f"[Source](SOURCE_{source_num})")
        
        return f"({', '.join(links)})"
    
    result = multi_source_pattern.sub(replace_multi_sources, text)
    
    single_source_pattern = re.compile(r'\[Source\]\(SOURCE_(\d+)\)')
    
    def replace_single_source(match):
        source_num = match.group(1)
        source_key = f"SOURCE_{source_num}"
        
        if source_key in src:
            url = src[source_key].get("url", "")
            if url:
                return f"[Source]({url})"
        
        return match.group(0)
    
    result = single_source_pattern.sub(replace_single_source, result)
    
    paren_single_pattern = re.compile(r'\(\[Source\]\(SOURCE_(\d+)\)\)')
    
    def replace_paren_single(match):
        source_num = match.group(1)
        source_key = f"SOURCE_{source_num}"
        
        if source_key in src:
            url = src[source_key].get("url", "")
            if url:
                return f"([Source]({url}))"
        
        return match.group(0)
    
    result = paren_single_pattern.sub(replace_paren_single, result)
    
    remaining_source_pattern = re.compile(r'SOURCE_(\d+)')
    
    def replace_remaining_source(match):
        source_num = match.group(1)
        source_key = f"SOURCE_{source_num}"
        
        if source_key in src:
            url = src[source_key].get("url", "")
            if url:
                return f"[Source]({url})"
        
        return match.group(0)
    
    result = remaining_source_pattern.sub(replace_remaining_source, result)
    
    return result

# ──────────────────────────────────────────────────────────────────────────────
# 9. Enhanced Memory Management - UPDATED FOR NEW CITATION SYSTEM
# ──────────────────────────────────────────────────────────────────────────────
def update_memory(mem: str, user: str, assistant: str) -> str:
    """Enhanced memory with better summarization"""
    clean_assistant = re.sub(r'\(\[Source\]\([^)]+\)\)', '', assistant)
    clean_assistant = re.sub(r'\[Source\]\([^)]+\)', '', clean_assistant)
    clean_assistant = re.sub(r'SOURCE_\d+', '', clean_assistant)
    clean_assistant = re.sub(r'\*\*Source:\*\*[^\n]*', '', clean_assistant)
    clean_assistant = clean_assistant.strip()
    
    new_exchange = f"\nUser: {user}\nPathOut: {clean_assistant}"
    draft = f"{mem}{new_exchange}"
    
    toks = draft.split()
    if len(toks) > MEMORY_MAX_TOK:
        toks = toks[-MEMORY_MAX_TOK:]
        draft = " ".join(toks)
        
        if "User:" in draft:
            start_idx = draft.find("User:")
            draft = draft[start_idx:]
    
    return draft

# ──────────────────────────────────────────────────────────────────────────────
# 10. Fixed Streamlit UI with Simplified Source Handling
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="👨‍⚕️ Pathology Specialist Assistant", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.session_state.setdefault("interrupt_request", False)
st.session_state.setdefault("processing", False)
st.session_state.setdefault("system_template", DEFAULT_SYSTEM_TMPL)

# Load vector store (moved here to ensure it's always loaded)
sig = compute_corpus_signature(ROOT_JSON_DIR)
index, chunks = load_vector_store(sig)

with st.sidebar:
    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.chat = []
        st.session_state.memory = ""
        st.session_state.interrupt_request = False
        st.session_state.processing = False
        st.rerun()
    
    if st.session_state.get("processing", False):
        if st.button("🛑 Stop Generation", use_container_width=True, type="primary"):
            st.session_state.interrupt_request = True
            st.warning("⏹️ Stopping response generation...")
            time.sleep(0.5)
            st.rerun()
    
    st.markdown("### 📝 Prompt Template Editor")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄", help="Reset to default template"):
            st.session_state.system_template = DEFAULT_SYSTEM_TMPL
            st.rerun()
    
    updated_template = st.text_area(
        "Edit system instructions:",
        value=st.session_state.system_template,
        height=300,
        help="Customize how the AI responds to questions. Changes take effect immediately.",
        label_visibility="collapsed"
    )
    
    if updated_template != st.session_state.system_template:
        st.session_state.system_template = updated_template
    
    if st.session_state.system_template != DEFAULT_SYSTEM_TMPL:
        st.info("✏️ Using custom template")
    else:
        st.success("📝 Using default template")

st.title("👨‍⚕️ Pathology Specialist Assistant")
st.markdown("*Powered by Algoscope*")

st.session_state.setdefault("chat", [])
st.session_state.setdefault("memory", "")

for i, m in enumerate(st.session_state.chat):
    with st.chat_message(m["role"]):
        st.markdown(m["content"], unsafe_allow_html=True)
        
        if m["role"] == "assistant":
            if "retrieval_info" in m:
                info = m["retrieval_info"]
                status_text = f"🔍 {info['method']} • {info['sources_count']} sources"
                if m.get("interrupted", False):
                    status_text += " • ⏹️ Response interrupted by user"
                st.caption(status_text)

q = st.chat_input("Ask a pathology question...", disabled=st.session_state.get("processing", False))
if q and not st.session_state.get("processing", False):
    st.session_state.interrupt_request = False
    st.session_state.processing = True
    
    st.session_state.chat.append({"role": "user", "content": q})
    
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_content = ""

        src = {}
        
        col1, col2 = st.columns([4, 1])
        with col2:
            stop_button_placeholder = st.empty()

    with st.spinner("Thinking..."):
        hits = []
        was_interrupted = False
        ans = ""
        try:
            # Retrieve with HyDE and expansion
            hits = retrieve(q, index, chunks)
            
            if st.session_state.get("interrupt_request", False):
                st.session_state.processing = False
                st.session_state.interrupt_request = False
                st.warning("⏹️ Request interrupted during retrieval.")
                st.rerun()
            
            prompt, src = build_enhanced_prompt(q, hits, st.session_state["memory"])
            
            if st.session_state.get("interrupt_request", False):
                st.session_state.processing = False
                st.session_state.interrupt_request = False
                st.warning("⏹️ Request interrupted during prompt preparation.")
                st.rerun()
            
            with stop_button_placeholder:
                if st.button("🛑 Stop", key="inline_stop", help="Stop response generation"):
                    st.session_state.interrupt_request = True
            
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=500,
                ),
                stream=True
            )
            
            for chunk in response:
                if st.session_state.get("interrupt_request", False):
                    break
                
                if chunk.text:
                    full_response_content += chunk.text
                    display_content = smart_linkify(full_response_content, src)
                    message_placeholder.markdown(display_content, unsafe_allow_html=True)
                    
            was_interrupted = st.session_state.get("interrupt_request", False)
            
            stop_button_placeholder.empty()
            
            if was_interrupted:
                ans = "*[Response generation was stopped]*"
                message_placeholder.markdown(ans, unsafe_allow_html=True)
            else:
                ans = smart_linkify(full_response_content, src)
                message_placeholder.markdown(ans, unsafe_allow_html=True)
            
        except Exception as e:
            ans = f"❌ Error generating response: {e}"
            was_interrupted = False
            message_placeholder.markdown(ans)
            stop_button_placeholder.empty()
        
        st.session_state.processing = False
        st.session_state.interrupt_request = False
        
        assistant_message = {
            "role": "assistant",
            "content": ans,
            "src": src,
            "interrupted": was_interrupted,
            "retrieval_info": {
                "method": "All information is obtained from https://www.pathologyoutlines.com",
                "sources_count": len(hits)
            }
        }
        st.session_state.chat.append(assistant_message)
        
        if not was_interrupted:
            st.session_state["memory"] = update_memory(
                st.session_state["memory"], q, full_response_content
            )
        
        if "retrieval_info" in assistant_message:
            info = assistant_message["retrieval_info"]
            status_text = f"🔍 {info['method']} • {info['sources_count']} sources"
            if was_interrupted:
                status_text += " • ⏹️ Response interrupted by user"
            st.caption(status_text)

st.markdown("---")
st.caption("PathOut • Powered by Algoscope • © 2025")
