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
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background-color: #1a1a1a !important;
    color: #e8e3d9 !important;
    font-family: 'Sora', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

[data-testid="stSidebar"] {
    background-color: #141414 !important;
    border-right: 1px solid #2a2a2a !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Sora', sans-serif !important;
    color: #b0a99e !important;
}

.main .block-container {
    max-width: 820px !important;
    padding: 2rem 1.5rem !important;
    margin: 0 auto !important;
}

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
.ai-label   { color: #82a8c4; }

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

/* ── MAIN area buttons — gold ── */
section[data-testid="stMain"] .stButton button {
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
section[data-testid="stMain"] .stButton button:hover {
    background: #d4b892 !important;
    transform: translateY(-1px) !important;
}

/* ── SIDEBAR history buttons — dark stealth ── */
div[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {
    background-color: #1e1e1e !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 10px !important;
    text-align: left !important;
    padding: 10px 14px !important;
    box-shadow: none !important;
    transition: border-color 0.15s, background-color 0.15s !important;
    width: 100% !important;
}
div[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover {
    border-color: #444444 !important;
    background-color: #222222 !important;
}
div[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] div[data-testid="stMarkdownContainer"] {
    color: #a39e93 !important;
    font-size: 0.82rem !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 400 !important;
}
div[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover div[data-testid="stMarkdownContainer"] {
    color: #ffffff !important;
}
div[data-testid="stSidebar"] .element-container {
    margin-bottom: 2px !important;
}

/* ── New Chat button stays gold ── */
div[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
    background: #c4a882 !important;
    color: #1a1a1a !important;
    border: none !important;
}

hr { border-color: #2a2a2a !important; }

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

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #141414; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }

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
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
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
    except Exception:
        return None

@st.cache_resource
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Missing Gemini API Key. Check your .env file setup.")
    return genai.Client(api_key=api_key)

def retrieve_chunks(query, n_results=5):
    model = load_model()
    collection = load_vectorstore()
    if collection is None:
        return None
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results

def ask_gemini(query, context, chat_history=""):
    gemini_client = get_gemini_client()
    prompt = f"""You are PaperMind, a research assistant that answers questions from scientific papers.

Rules:
- Answer ONLY from the context below
- Use the CONVERSATION HISTORY to understand follow-up references
- Be precise and cite which paper each fact comes from
- If the answer is not in the context say: "I couldn't find this in the downloaded papers."
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
            if response and response.text:
                return response.text
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                return f"Error connecting to model interface layer: {str(e)}"
    return "Error generating response. Please try again."

def run_rag(question):
    results = retrieve_chunks(question, n_results=5)
    if not results or not results.get("documents") or len(results["documents"][0]) == 0:
        return "Vector store chunks not found matching query parameters. Check if your documents have been processed.", []

    context_parts = []
    sources = []
    seen = set()

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(f"[{meta['title'][:50]}]\n{doc}")
        if meta["title"] not in seen:
            sources.append(meta["title"])
            seen.add(meta["title"])

    context = "\n---\n".join(context_parts)

    chat_history_str = ""
    for msg in st.session_state.messages[-4:]:
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

    collection = load_vectorstore()
    if collection:
        count = collection.count()
        st.markdown(
            f'<span class="status-ready">● {count} chunks indexed</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-empty">● No vector store found</span>',
            unsafe_allow_html=True
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    st.markdown("**Research Topic**")
    topic = st.text_input(
        "",
        value=st.session_state.current_topic,
        placeholder="e.g. Graphene doping",
        label_visibility="collapsed"
    )
    st.session_state.current_topic = topic

    st.markdown("---")
    st.markdown("**Past Sessions**")
    history = load_memory()

    if history:
        for entry in reversed(history[-8:]):
            short_q = entry["question"][:35] + ("..." if len(entry["question"]) > 35 else "")
            btn_label = f"🕒 {entry['timestamp'][5:10]} | {short_q}"

            if st.button(
                btn_label,
                key=f"hist_btn_{entry['id']}",
                help=entry["question"],
                use_container_width=True
            ):
                st.session_state.messages = [
                    {"role": "user", "content": entry["question"]},
                    {"role": "assistant", "content": entry["answer"], "sources": entry.get("sources", [])}
                ]
                st.session_state.current_topic = entry.get("topic", "General")
                st.rerun()
    else:
        st.markdown(
            '<div style="color:#444;font-size:0.82rem">No history yet.</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    if history:
        st.metric("Total Questions", len(history))

    if st.button("New Chat", use_container_width=True, key="new_chat_btn"):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1.5rem">
    <div style="font-size:2rem;margin-bottom:8px">🧠</div>
    <h1 style="font-size:1.6rem;font-weight:600;color:#e8e3d9;margin:0">PaperMind</h1>
    <p style="color:#555;font-size:0.88rem;margin-top:6px">Ask questions from your research papers</p>
</div>
""", unsafe_allow_html=True)

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
            Try: "What methods are used to study graphene doping?"<br/>
            Or: "Summarize the key findings on formation energy prediction"
        </div>
    </div>
    """, unsafe_allow_html=True)

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
            # Fallback to get keys safely if sources structure is modified
            for s in msg.get("sources", [])
        ])
        st.markdown(f"""
        <div class="ai-msg">
            <div class="msg-label ai-label">PaperMind</div>
            {msg["content"].replace(chr(10), "<br>")}
            <div style="margin-top:12px">{sources_html}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT FORM LOOP (Handles Enter-Key Submission)
# ─────────────────────────────────────────────
st.markdown("<br/>", unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    with col1:
        question = st.text_input(
            "",
            placeholder="Ask a question about your papers...",
            label_visibility="collapsed",
            key="question_input"
        )
    with col2:
        send = st.form_submit_button("Ask →", use_container_width=True)

if send and question.strip():
    # Append user question to prompt tracking history
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    
    # Process retrieval loops directly
    with st.spinner("Searching papers..."):
        answer, sources = run_rag(question)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

    add_to_memory(question, answer, sources, st.session_state.current_topic)
    st.rerun()