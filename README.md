<div align="center">

# 🧠 PaperMind RAG

### *Turn research papers into answers — instantly.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-orange?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-green?style=for-the-badge)](https://www.trychroma.com)
[![arXiv](https://img.shields.io/badge/arXiv-Paper_Source-red?style=for-the-badge)](https://arxiv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Status](https://img.shields.io/badge/Status-Active_Development-brightgreen?style=for-the-badge)]()

<br/>


Added Few Research Content

<br/>
> **Describe your research → auto-fetch papers → ask questions → get cited answers.**
>
> PaperMind is an end-to-end RAG pipeline built from scratch. It fetches domain-specific research papers from arXiv, parses and chunks them intelligently, embeds them into a vector store, and answers your questions — grounded in real papers, not hallucinations. Built with a persistent memory layer and a clean chat UI.

<br/>

```
📝 Research Description
        ↓
🔑 AI Keyword Extraction (Gemini)
        ↓
📡 arXiv Fetch + PDF Download
        ↓
📄 Parse → Clean → Chunk (pymupdf4llm)
        ↓
🧬 Embed → ChromaDB Vector Store
        ↓
💬 Query → Retrieve → Answer (with citations)
        ↓
🧠 Persistent Memory Layer
```

</div>

---

## ✅ What's Built

| # | File | What it does | Status |
|---|------|-------------|--------|
| 1 | `1_context_Extractor_pipeline.py` | User describes research → Gemini extracts 25+ arXiv keywords | ✅ Working |
| 2 | `paper_downloader.py` | Batched OR queries → rate-limited arXiv fetch → PDF download | ✅ Working |
| 3 | `3_pdf_parser.py` | pymupdf4llm parsing → noise cleaning → section-aware chunking | ✅ Working |
| 4 | `4_embedder.py` | SentenceTransformer embeddings → ChromaDB persistent store | ✅ Working |
| 5 | `5_rag_query.py` | Query → top-K retrieval → Gemini answer with paper citations | ✅ Working |
| 6 | `app.py` | Streamlit chat UI with memory layer and session history | ✅ Working |
| — | `Memory Layer` | JSON-based persistent chat history across all sessions | ✅ Working |

**Current stats:** 6 papers · 225 chunks · 384-dim embeddings · persistent vector store

---

## 🏗️ Architecture

```
```text
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██████╗  █████╗ ██████╗ ███████╗██████╗ ███╗   ███╗██╗███╗   ██╗██████╗    ║
║   ██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗   ║
║   ██████╔╝███████║██████╔╝█████╗  ██████╔╝██╔████╔██║██║██╔██╗ ██║██║  ██║   ║
║   ██╔═══╝ ██╔══██║██╔═══╝ ██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║╚██╗██║██║  ██║   ║
║   ██║     ██║  ██║██║     ███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██████╔╝   ║
║   ╚═╝     ╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝    ║
║                                                                              ║
║                    🧠 AI-Powered Research Retrieval Engine                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌────────────────────┐        ┌────────────────────┐                        ║
║  │    📥 INGESTION     │        │     💾 STORAGE      │                     ║
║  ├────────────────────┤        ├────────────────────┤                        ║
║  │ • Keyword Parsing  │ ─────▶ │ • PDFs on Disk     │                       ║
║  │ • arXiv Fetching   │        │ • chunks.json      │                        ║
║  │ • PDF Extraction   │ ─────▶ │ • ChromaDB         │                       ║
║  │ • Text Chunking    │        │ • Vector Store     │                        ║
║  └────────────────────┘        └────────────────────┘                        ║
║               │                             ▲                                ║
║               │                             │                                ║
║               ▼                             │                                ║ 
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │                           🔍 QUERY ENGINE                            │    ║
║  ├──────────────────────────────────────────────────────────────────────┤    ║
║  │  User Question                                                       │   ║
║  │        ↓                                                             │   ║
║  │  Query Embedding                                                     │   ║
║  │        ↓                                                             │   ║
║  │  Cosine Similarity Search                                            │   ║
║  │        ↓                                                             │   ║
║  │  Retrieve Top-K Relevant Chunks                                      │   ║
║  │        ↓                                                             │   ║
║  │  Gemini-Powered Response Generation                                  │   ║
║  │        ↓                                                             │   ║
║  │  Source Attribution & Citation                                       │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
║                                                                             ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │                         🧠 MEMORY LAYER                              │  ║
║  ├──────────────────────────────────────────────────────────────────────┤   ║
║  │  data/memory/chat_history.json                                       │   ║
║  │                                                                      │   ║
║  │   • Timestamp Tracking                                               │   ║
║  │   • Session Persistence                                              │   ║
║  │   • Topic Awareness                                                  │   ║
║  │   • Question / Answer Logging                                        │   ║
║  │   • Source Reference Storage                                         │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Gemini API Key — free at [aistudio.google.com](https://aistudio.google.com/apikey)

### Installation

```bash
git clone https://github.com/amaanarif755/PaperMind-RAG.git
cd PaperMind-RAG
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env
```

### Run the Pipeline

```bash
# 1. Fetch papers
python Main/1_context_Extractor_pipeline.py

# 2. Parse PDFs
python Main/3_pdf_parser.py

# 3. Build vector store
python Main/4_embedder.py

# 4. Launch UI
streamlit run app.py
```

---

## 💬 Demo

```
┌─────────────────────────────────────────────────────────┐
│  🧠 PaperMind                        ● 225 chunks indexed│
│─────────────────────────────────────────────────────────│
│                                                         │
│  You                                                    │
│  ┌────────────────────────────────────────────────────┐ │
│  │ What methods predict graphene formation energy?    │ │
│  └────────────────────────────────────────────────────┘ │
│                                                         │
│  PaperMind                                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Raman spectroscopy provides insight into graphene  │ │
│  │ mobility and doping via G and 2D band shifts.      │ │
│  │                                                    │ │
│  │ 📄 A Chemical Route to Graphene...                 │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **LLM** | Google Gemini 2.5 Flash | Free API, fast, good at citing |
| **Embeddings** | `all-MiniLM-L6-v2` | 90MB, fast, good semantic similarity |
| **Vector Store** | ChromaDB (persistent) | Local, no cloud needed, fast retrieval |
| **PDF Parsing** | pymupdf4llm | 2025 best practice for LLM-ready markdown output |
| **Paper Source** | arXiv API | Free, 2M+ papers, no auth needed |
| **UI** | Streamlit | Fast to build, easy to demo |
| **Memory** | JSON (persistent) | Simple, portable, human-readable history |

---

## 🗺️ Roadmap

### ✅ Phase 1 — Core Pipeline (Complete)
- [x] AI keyword extraction from research description
- [x] arXiv paper fetching with rate limiting + deduplication
- [x] Section-aware PDF chunking with pymupdf4llm
- [x] Semantic embeddings + persistent ChromaDB vector store
- [x] RAG query with cited answers
- [x] Streamlit chat UI
- [x] Persistent memory layer — chat history across sessions

### 🔨 Phase 2 — Intelligence Layer (Next)
- [ ] Query expansion — decompose complex questions into sub-queries
- [ ] Re-ranking — score retrieved chunks by relevance before LLM call
- [ ] Conversation context — use previous Q&A pairs to improve follow-up answers
- [ ] Multi-hop reasoning — chain answers across multiple papers

### 🔬 Phase 3 — Research Features
- [ ] Citation graph — surface papers cited by your papers
- [ ] Paper summariser — one-click abstract + key contributions
- [ ] Export to Notion / Markdown research notes
- [ ] Semantic similarity between papers — find related work automatically

### 🚀 Phase 4 — Scale
- [ ] Semantic Scholar + PubMed as additional paper sources
- [ ] Upload your own PDFs (not just arXiv)
- [ ] Multi-user support with separate knowledge bases
- [ ] Evaluation dashboard — RAGAS metrics for retrieval quality

---

## 📁 Project Structure

```
PaperMind-RAG/
├── Main/
│   ├── 1_context_Extractor_pipeline.py  # Keyword extraction
│   ├── paper_downloader.py              # arXiv fetcher
│   ├── 3_pdf_parser.py                  # PDF → chunks
│   ├── 4_embedder.py                    # Embeddings + ChromaDB
│   └── 5_rag_query.py                   # Terminal RAG interface
├── app.py                               # Streamlit UI + memory
├── data/
│   ├── papers/                          # PDFs + metadata (gitignored)
│   ├── vectorstore/                     # ChromaDB (gitignored)
│   └── memory/                          # Chat history JSON
├── .env                                 # API keys (gitignored)
├── requirements.txt
└── README.md
```

---

## 🙋 About

Built by **[Amaan Arif](https://www.linkedin.com/in/amaanarif755)** — 2nd year B.Tech Chemical Engineering @ MNNIT Allahabad.

This project is a from-scratch implementation of a domain-specific RAG system, built as part of a deep dive into LLMs, retrieval systems, and enterprise AI. Every component — keyword extraction, paper fetching, parsing, embedding, retrieval — is built and understood independently.

---

<div align="center">

*Built from scratch. Every bug fixed manually. Every component understood.*

⭐ Star if useful · 🍴 Fork to extend · 📬 Reach out to collaborate

</div>
