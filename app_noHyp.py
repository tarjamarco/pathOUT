#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
# Enhanced Pathology Chatbot · RAG with Query Expansion and improved retrieval
# ──────────────────────────────────────────────────────────────────────────────
import os, json, pickle, hashlib, textwrap, re, types
import streamlit as st
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
import re
from typing import List, Dict, Tuple, Any
import time
import threading
import concurrent.futures # Add for parallel execution
import nltk

# Only download if not already present
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
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
ROOT_JSON_DIR   = "new_json"
EMBED_MODEL_ID  = "neuml/pubmedbert-base-embeddings"
CROSS_ENCODER_ID= "ncbi/MedCPT-Cross-Encoder"
INDEX_PATH      = "faiss.idx"
META_PATH       = "chunk_meta.pkl"
CHUNK_TOKENS    = 140
CHUNK_OVERLAP   = 30
TOP_K_FAST      = 50         # Reduced as HyDE is removed
TOP_K_FINAL     = 5          # Recommended final sources
MEMORY_MAX_TOK  = 1000

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ──────────────────────────────────────────────────────────────────────────────
# 2. Utility functions (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
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
# 3. Enhanced Query Processing (Modified - Removed HyDE)
# ──────────────────────────────────────────────────────────────────────────────
# Removed generate_hypothetical_document

def expand_query(question: str) -> List[str]:
    """Generate query variations and synonyms"""
    if st.session_state.get("interrupt_request", False):
        return [question]
        
    expansion_prompt = f"""
Generate 3-4 alternative ways to ask this pathology question, using different medical terminology and phrasing. Focus on:
- Synonyms for key terms
- Different clinical contexts
- Alternative diagnostic approaches

Original question: {question}

Provide only the alternative questions, one per line:
"""
    
    try:
        # Use a faster model for expansion if possible
        response = genai.GenerativeModel("gemini-2.0-flash").generate_content(expansion_prompt)
        alternatives = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
        return [question] + alternatives[:3]  # Original + top 3 alternatives
    except Exception as e:
        st.warning(f"Query expansion failed: {e}")
        return [question]

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
    out = []
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

        for ch_text in sentence_chunks(sec.get("clean_content", "")):
            out.append({
                "text": ch_text,
                "topic": sec.get("topic", ""),
                "url": sec.get("url", ""),
                "actual_json_file_path": actual_json_file_rel_path,
                "display_source_id": display_source_id,
            })
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
# 6. Enhanced Retrieval with Query Expansion (Modified - Removed HyDE)
# ──────────────────────────────────────────────────────────────────────────────
def retrieve_with_expansion(question: str, index, chunks) -> List[Tuple[float, Dict]]:
    """Enhanced retrieval using query expansion and cross-encoding"""
    if st.session_state.get("interrupt_request", False):
        return []
        
    all_candidates = {}
    
    # Run query expansion in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_expanded_queries = executor.submit(expand_query, question)
        
        # Original query retrieval (can run immediately)
        original_candidates = retrieve_basic(question, index, chunks, TOP_K_FAST)
        for score, chunk in original_candidates:
            chunk_id = id(chunk)
            if chunk_id not in all_candidates or score > all_candidates[chunk_id][0]:
                all_candidates[chunk_id] = (score, chunk)
        
        # Wait for expanded queries
        expanded_queries = future_expanded_queries.result()
        if st.session_state.get("interrupt_request", False):
            return list(all_candidates.values())[:TOP_K_FINAL]
            
        for exp_query in expanded_queries[1:]:  # Skip the original query
            if st.session_state.get("interrupt_request", False):
                break
            # Retrieve with a slightly lower TOP_K for expanded queries as they are secondary
            exp_candidates = retrieve_basic(exp_query, index, chunks, TOP_K_FAST // 2)
            for score, chunk in exp_candidates:
                chunk_id = id(chunk)
                # Weight expanded queries lower
                adjusted_score = score * 0.8
                if chunk_id not in all_candidates or adjusted_score > all_candidates[chunk_id][0]:
                    all_candidates[chunk_id] = (adjusted_score, chunk)
    
    candidate_list = list(all_candidates.values())
    
    if st.session_state.get("interrupt_request", False):
        return candidate_list[:TOP_K_FINAL]
    
    # Re-rank with cross-encoder
    pairs = [[question, chunk["text"]] for _, chunk in candidate_list]
    if pairs:
        cross_scores = cross_encoder.predict(pairs, batch_size=32)
        
        # Combine semantic similarity and cross-encoder scores
        combined_scores = []
        for i, (sem_score, chunk) in enumerate(candidate_list):
            combined_score = 0.6 * cross_scores[i] + 0.4 * sem_score # Weighted average
            combined_scores.append((combined_score, chunk))
        
        # Sort by combined score and return top results
        combined_scores.sort(key=lambda x: x[0], reverse=True)
        return combined_scores[:TOP_K_FINAL]
    
    return []

def retrieve_basic(query: str, index, chunks, top_k: int = TOP_K_FAST) -> List[Tuple[float, Dict]]:
    """Basic retrieval without re-ranking"""
    qv = embedder_single.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(qv, min(top_k, len(chunks)))
    return [(float(scores[0][i]), chunks[ids[0][i]]) for i in range(len(ids[0]))]

def retrieve(question: str, index, chunks):
    """Main retrieval function - now uses query expansion only"""
    return retrieve_with_expansion(question, index, chunks)

# ──────────────────────────────────────────────────────────────────────────────
# 7. Enhanced Prompt Building - UPDATED FOR CONCISE RESPONSES
# ──────────────────────────────────────────────────────────────────────────────

FEW_SHOTS = """
### Example A1
Acute promyelocytic leukaemia (APL) is characterised by abnormal promyelocytes with heavy granulation and the pathognomonic t(15;17) translocation creating PML-RARA fusion ([Source](https://www.pathologyoutlines.com/topic/example1.html)). Auer rods are frequently present in malignant cells, and there is a high risk of DIC due to procoagulant release ([Source](https://www.pathologyoutlines.com/topic/example2.html)).
"""

# Default system template - UPDATED TO INSTRUCT PROPER CITATION FORMAT
DEFAULT_SYSTEM_TMPL = textwrap.dedent("""
You are PathOut, an AI assistant from Pathology Outlines, developed by Algoscope, created to support medical professionals—including pathology residents, physicians, and lab technologists—with accurate, concise answers.                                      

Based EXCLUSIVELY on the following text compilation from various documents AND the conversation history if provided, answer the user's latest question. Assume the user is a medical professional who requires specific, technical details. Do not use any prior knowledge outside the context.
Respect the instruction but don't begin your response with "Based on the provided text", begin directly with the response.
If asked about your identity, state that you are PathOut, an AI assistant from Pathology Outlines and developed by Algoscope. Do not mention Google or being a large language model.

**CRITICAL Citation Instructions:** 
When referencing specific information from the sources, you MUST use this EXACT format:
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
    
    retrieval_info = "Advanced retrieval with Query Expansion"
    
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
# 8. FIXED linkification - HANDLES MULTIPLE SOURCE CITATIONS
# ──────────────────────────────────────────────────────────────────────────────
def linkify(text: str, src: dict) -> str:
    """Convert SOURCE_N references to clickable links with proper URL handling"""
    
    # Pattern 1: Handle multiple sources in parentheses like ([Source](SOURCE_1), [Source](SOURCE_2), [Source](SOURCE_3))
    multi_source_pattern = re.compile(r'\((\[Source\]\(SOURCE_\d+\)(?:,\s*\[Source\]\(SOURCE_\d+\))*)\)')
    
    def replace_multi_sources(match):
        # Extract the content inside parentheses
        sources_content = match.group(1)
        
        # Find all individual SOURCE_N references
        individual_sources = re.findall(r'\[Source\]\(SOURCE_(\d+)\)', sources_content)
        
        # Convert each source to a proper link
        links = []
        for source_num in individual_sources:
            source_key = f"SOURCE_{source_num}"
            if source_key in src:
                url = src[source_key].get("url", "")
                if url:
                    links.append(f"[Source]({url})")
                else:
                    links.append(f"[Source](SOURCE_{source_num})")
            else:
                links.append(f"[Source](SOURCE_{source_num})")
        
        # Return formatted with parentheses
        return f"({', '.join(links)})"
    
    # Apply multi-source replacement
    result = multi_source_pattern.sub(replace_multi_sources, text)
    
    # Pattern 2: Handle single source citations ([Source](SOURCE_N))
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
    
    # Pattern 3: Handle parenthesized single sources ([Source](SOURCE_N))
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
    
    # Pattern 4: Handle any remaining SOURCE_N references
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
# 9. Enhanced Memory Management - UPDATED TO HANDLE NEW CITATION FORMAT
# ──────────────────────────────────────────────────────────────────────────────
def update_memory(mem: str, user: str, assistant: str) -> str:
    """Enhanced memory with better summarization"""
    # Remove citations from memory to keep it clean
    clean_assistant = re.sub(r'\(\[Source\]\([^)]+\)\)', '', assistant)
    clean_assistant = re.sub(r'\[Source\]\([^)]+\)', '', clean_assistant)
    clean_assistant = re.sub(r'SOURCE_\d+', '', clean_assistant)
    
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
# 10. Fixed Streamlit UI with Proper Streaming
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
        
        col1, col2 = st.columns([4, 1])
        with col2:
            stop_button_placeholder = st.empty()

    with st.spinner("Thinking..."):
        try:
            # Retrieve without HyDE
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
            
            # --- STREAMING WITH PROPER LINKIFICATION ---
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=500,
                ),
                stream=True
            )
            
            # Stream response and accumulate text
            for chunk in response:
                if st.session_state.get("interrupt_request", False):
                    break
                
                if chunk.text:
                    full_response_content += chunk.text
                    # Apply linkify to current content and display
                    display_content = linkify(full_response_content, src)
                    message_placeholder.markdown(display_content, unsafe_allow_html=True)
                    
            was_interrupted = st.session_state.get("interrupt_request", False)
            
            stop_button_placeholder.empty()
            
            if was_interrupted:
                ans = "*[Response generation was stopped]*"
                message_placeholder.markdown(ans, unsafe_allow_html=True)
            else:
                # Final linkification of the complete response
                ans = linkify(full_response_content, src)
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
