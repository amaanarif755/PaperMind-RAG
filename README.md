<div align="center">

# 🧠 PaperMind RAG

### *Turn research papers into answers — instantly.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-orange?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-green?style=for-the-badge)](https://www.trychroma.com)
[![arXiv](https://img.shields.io/badge/arXiv-Paper_Source-red?style=for-the-badge)](https://arxiv.org)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

<br/>

> **Describe your research → get papers → ask questions → get cited answers.**
> 
> PaperMind is an end-to-end RAG (Retrieval-Augmented Generation) pipeline that automatically fetches domain-specific research papers from arXiv, parses and chunks them, embeds them into a vector store, and lets you query them with an LLM — all grounded in real papers, not hallucinations.

<br/>

```
📝 Describe Research  →  🔑 Extract Keywords  →  📥 Fetch Papers  →  📄 Parse & Chunk  →  🧬 Embed  →  💬 Ask & Get Answers
```

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **AI Keyword Extraction** | Gemini 2.5 Flash generates 25+ domain-specific arXiv search keywords from your research description |
| 📡 **Smart Paper Fetching** | Batched OR queries with exponential backoff — respects arXiv rate limits, deduplicates results |
| 📄 **Research-Grade Parsing** | `pymupdf4llm` extracts clean Markdown from PDFs, preserving tables and section headers |
| 🧬 **Semantic Embeddings** | `all-MiniLM-L6-v2` encodes 225+ chunks into a persistent ChromaDB vector store |
| 💬 **Grounded Q&A** | Answers come from your papers only — every response cites the source paper |
| 🔁 **Incremental Updates** | Re-run the pipeline anytime; skips already-downloaded papers automatically |

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PaperMind Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. context_Extractor_pipeline.py                               │
│     User Input → Gemini 2.5 Flash → 25+ Keywords               │
│                        │                                        │
│                        ▼                                        │
│  2. paper_downloader.py                                         │
│     Keywords → Batched arXiv Queries → PDFs + Metadata JSON     │
│                        │                                        │
│                        ▼                                        │
│  3. pdf_parser.py                                               │
│     PDFs → pymupdf4llm → Clean Text → Section Chunks           │
│                        │                                        │
│                        ▼                                        │
│  4. embedder.py                                                 │
│     Chunks → SentenceTransformer → ChromaDB Vector Store        │
│                        │                                        │
│                        ▼                                        │
│  5. rag_query.py                                                │
│     Question → Embed → Retrieve Top-K → Gemini → Cited Answer  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Gemini API Key (free at [aistudio.google.com](https://aistudio.google.com/apikey))

### Installation

```bash
# Clone the repo
git clone https://github.com/amaanarif755/PaperMind-RAG.git
cd PaperMind-RAG

# Install dependencies
pip install -r requirements.txt

# Set up environment
echo "GEMINI_API_KEY=your_key_here" > .env
```

### Run the Full Pipeline

```bash
# Step 1: Extract keywords + download papers
python Main/1_context_Extractor_pipeline.py

# Step 2: Parse PDFs into chunks
python Main/3_pdf_parser.py

# Step 3: Build vector store
python Main/4_embedder.py

# Step 4: Ask questions
python Main/5_rag_query.py
```

---

## 💬 Demo

```
=== PaperMind RAG System ===

Describe your research: doping of graphene sheets and predicting formation energy

✅ Keywords extracted: 30
✅ Papers downloaded: 6
✅ Chunks created: 225
✅ Vector store: 225 chunks indexed

Your question: What methods are used to study graphene electronic properties?

🔍 Retrieving relevant chunks...
📄 Source: A Chemical Route to Graphene for Electronics and Spintronics

Answer:
Raman spectroscopy provides insight into mobility of graphene devices and 
information about types and degree of doping based on observed shifts of 
G and 2D bands. [Paper: A Chemical Route to Graphene for Electronics and Spintronics]
```

---

## 📁 Project Structure

```
PaperMind-RAG/
├── Main/
│   ├── 1_context_Extractor_pipeline.py  # Keyword extraction via Gemini
│   ├── paper_downloader.py              # arXiv scraper + PDF downloader
│   ├── 3_pdf_parser.py                  # PDF parsing + chunking
│   ├── 4_embedder.py                    # Embedding + ChromaDB indexing
│   └── 5_rag_query.py                   # RAG query interface
├── data/
│   ├── papers/                          # Downloaded PDFs (gitignored)
│   └── vectorstore/                     # ChromaDB persistent store
├── .env                                 # API keys (gitignored)
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Google Gemini 2.5 Flash |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | ChromaDB (persistent) |
| **PDF Parsing** | pymupdf4llm + PyMuPDF |
| **Paper Source** | arXiv API |
| **Language** | Python 3.10+ |

---

## 📊 Current Stats

```
Papers downloaded   : 6
Total chunks        : 225
Embedding model     : all-MiniLM-L6-v2 (90MB)
Vector dimensions   : 384
Chunking strategy   : Section-based with 100-char overlap
```

---

## 🗺️ Roadmap

- [x] Keyword extraction via LLM
- [x] arXiv paper fetching with rate limiting
- [x] PDF parsing with section-aware chunking
- [x] Semantic embeddings + vector store
- [x] RAG query with cited answers
- [ ] Streamlit web UI
- [ ] Query expansion for better retrieval
- [ ] Multi-hop reasoning across papers
- [ ] Citation graph integration
- [ ] Export answers as research notes

---

## 🙋 About

Built by **[Amaan Arif](https://www.linkedin.com/in/amaanarif755)** — 2nd year B.Tech Chemical Engineering student at MNNIT Allahabad, exploring the intersection of AI systems and research automation.

This project is part of a broader learning journey into RAG systems, LLMs, and enterprise AI.

---

<div align="center">

⭐ **Star this repo if you find it useful** ⭐

</div>