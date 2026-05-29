import streamlit as st
import json
import os
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PaperMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Claude-inspired dark UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background-color: #1a1a1a !important;
    color: #e8e3d9 !important;
    font-family: 'Sora', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #141414 !important;
    border-right: 1px solid #2a2a2a !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Sora', sans-serif !important;
    color: #b0a99e !important;
}

/* ── Main container ── */
.main .block-container {
    max-width: 820px !important;
    padding: 2rem 1.5rem !important;
    margin: 0 auto !important;
}

/* ── Chat messages ── */
.user-msg {
    background: #2a2a2a;
    border: 1px solid #333;
    border-radius: 18px 18px 4px 18px;
    padding: 14px 18px;
    margin: 12px 0 12px 60px;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #e8e3d9;
}

.ai-msg {
    background: #1e1e1e;
    border: 1px solid #2e2e2e;
    border-radius: 18px 18px 18px 4px;
    padding: 16px 20px;
    margin: 12px 60px 12px 0;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #d4cfc6;
}

.msg-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
    opacity: 0.5;
}

.user-label { color: #c4a882; }
.ai-label { color: #82a8c4; }

/* ── Source chips ── */
.source-chip {
    display: inline-block;
    background: #252525;
    border: 1px solid #383838;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem;
    color: #888;
    margin: 4px 4px 0 0;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Input area ── */
.stTextInput input, .stTextArea textarea {
    background: #222 !important;
    border: 1px solid #333 !important;
    border-radius: 12px !important;
    color: #e8e3d9 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.93rem !important;
    padding: 12px 16px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #c4a882 !important;
    box-shadow: 0 0 0 2px rgba(196,168,130,0.12) !important;
}

/* ── Buttons ── */
.stButton button {
    background: #c4a882 !important;
    color: #1a1a1a !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px 22px !important;
    transition: all 0.2s ease !important;
}
.stButton button:hover {
    background: #d4b892 !important;
    transform: translateY(-1px) !important;
}

/* ── Dividers ── */
hr { border-color: #2a2a2a !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 12px 16px !important;
}
[data-testid="stMetricValue"] {
    color: #c4a882 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.6rem !important;
}
[data-testid="stMetricLabel"] {
    color: #666 !important;
    font-size: 0.78rem !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #1e1e1e !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 10px !important;
    color: #888 !important;
    font-size: 0.82rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #141414; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }

/* ── Status badge ── */
.status-ready {
    display: inline-block;
    background: rgba(130,196,130,0.12);
    border: 1px solid rgba(130,196,130,0.25);
    color: #82c482;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.status-empty {
    display: inline-block;
    background: rgba(196,168,130,0.12);
    border: 1px solid rgba(196,168,130,0.25);
    color: #c4a882;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
}

/* ── History item ── */
.history-item {
    background: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    cursor: pointer;
    font-size: 0.83rem;
    color: #888;
    line-height: 1.4;
    transition: border-color 0.15s;
}
.history-item:hover { border-color: #444; color: #aaa; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MEMORY LAYER
# ─────────────────────────────────────────────
MEMORY_FILE = "data/memory/chat_history.json"

def load_memory():
    os.makedirs("data/memory", exist_ok=True)
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(history):
    os.makedirs("data/memory", exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def add_to_memory(question, answer, sources, topic):
    history = load_memory()
    history.append({
        "id": len(history) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "topic": topic,
        "question": question,
        "answer": answer,
        "sources": sources
    })
    save_memory(history)
    return history

# ─────────────────────────────────────────────
# RAG CORE
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_vectorstore():
    try:
        client = chromadb.PersistentClient(path="data/vectorstore")
        collection = client.get_collection("papers")
        return collection
    except:
        return None

def retrieve_chunks(query, n_results=5):
    model = load_model()
    collection = load_vectorstore()
    if not collection:
        return None
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results

def ask_gemini(query, context, chat_history=""):
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = f"""You are PaperMind, a research assistant that answers questions from scientific papers.

Rules:
- Answer ONLY from the context below
- Use the CONVERSATION HISTORY to understand references (like "what did you mean by that?")
- Be precise and cite which paper each fact comes from
- If not in context, say: "I couldn't find this in the downloaded papers."
- Format your answer clearly with line breaks

CONVERSATION HISTORY:
{chat_history}

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    for attempt in range(3):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
    return "Error generating response. Please try again."

def run_rag(question):
    results = retrieve_chunks(question, n_results=5)
    if not results:
        return "Vector store not found. Run the pipeline first.", []

    context_parts = []
    sources = []
    seen = set()
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(f"[{meta['title'][:50]}]\n{doc}")
        if meta["title"] not in seen:
            sources.append(meta["title"])
            seen.add(meta["title"])

    context = "\n---\n".join(context_parts)
    
    # Extract Conversation History for follow-ups
    chat_history_str = ""
    recent_messages = st.session_state.messages[-4:] # Grab the last 4 exchanges
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "PaperMind"
        chat_history_str += f"{role}: {msg['content']}\n"

    answer = ask_gemini(question, context, chat_history_str)
    return answer, sources

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_topic" not in st.session_state:
    st.session_state.current_topic = "General"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 PaperMind")
    st.markdown("---")

    # Vector store status
    collection = load_vectorstore()
    if collection:
        count = collection.count()
        st.markdown(f'<span class="status-ready">● {count} chunks indexed</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-empty">● No vector store found</span>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Topic tag
    st.markdown("**Research Topic**")
    topic = st.text_input("", value=st.session_state.current_topic,
                          placeholder="e.g. Graphene doping",
                          label_visibility="collapsed")
    st.session_state.current_topic = topic

    st.markdown("---")

    # Chat history from memory
    st.markdown("**Past Sessions**")
    history = load_memory()

    if history:
        # Show last 8 entries
        for entry in reversed(history[-8:]):
            st.markdown(f"""
            <div class="history-item">
                <div style="color:#555;font-size:0.68rem;margin-bottom:3px">{entry['timestamp']} · {entry['topic']}</div>
                {entry['question'][:60]}{'...' if len(entry['question']) > 60 else ''}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#444;font-size:0.82rem">No history yet.</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # Stats
    if history:
        st.metric("Total Questions", len(history))

    # Clear current chat
    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────

# Header
st.markdown("""
<div style="text-align:center;padding:2rem 0 1.5rem">
    <div style="font-size:2rem;margin-bottom:8px">🧠</div>
    <h1 style="font-size:1.6rem;font-weight:600;color:#e8e3d9;margin:0">PaperMind</h1>
    <p style="color:#555;font-size:0.88rem;margin-top:6px">Ask questions from your research papers</p>
</div>
""", unsafe_allow_html=True)

# Empty state
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;color:#444">
        <div style="font-size:3rem;margin-bottom:1rem;opacity:0.3">📄</div>
        <div style="font-size:0.9rem;line-height:1.8">
            Ask anything about your downloaded papers.<br/>
            Answers are grounded in real research — not hallucinations.
        </div>
        <br/>
        <div style="font-size:0.78rem;color:#333">
            Try: "What methods are used to study graphene doping?" <br/>
            Or: "Summarize the key findings on formation energy prediction"
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-msg">
            <div class="msg-label user-label">You</div>
            {msg["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        sources_html = "".join([
            f'<span class="source-chip">📄 {s[:45]}</span>'
            for s in msg.get("sources", [])
        ])
        st.markdown(f"""
        <div class="ai-msg">
            <div class="msg-label ai-label">PaperMind</div>
            {msg["content"].replace(chr(10), '<br>')}
            <div style="margin-top:12px">{sources_html}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────
st.markdown("<br/>", unsafe_allow_html=True)

col1, col2 = st.columns([6, 1])

with col1:
    question = st.text_input(
        "",
        placeholder="Ask a question about your papers...",
        label_visibility="collapsed",
        key="question_input"
    )

with col2:
    send = st.button("Ask →", use_container_width=True)

# ─────────────────────────────────────────────
# PROCESS QUESTION
# ─────────────────────────────────────────────
if send and question.strip():
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Run RAG
    with st.spinner("Searching papers..."):
        answer, sources = run_rag(question)

    # Add AI message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

    # Save to persistent memory
    add_to_memory(question, answer, sources, st.session_state.current_topic)

    st.rerun()