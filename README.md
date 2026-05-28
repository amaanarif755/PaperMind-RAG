Heyyy Youuuuu \
This is project is kinda something i really wanna bring it to life... \
Learn, read and code thats the main objective of this particular Repo, will be experimenting with the stuff and exploring...

------------------------------------------------------------------------------------------------------------------------------
# PaperMind RAG
------------------------------------------------------------------------------------------------------------------------------

A RAG system that helps researchers find answers from domain-specific papers fetched from arXiv.

------------------------------------------------------------------------------------------------------------------------------
## Pipeline
------------------------------------------------------------------------------------------------------------------------------
1. User describes research → Gemini extracts keywords
2. Keywords → arXiv search → PDF download
3. PDFs → parsed and chunked (in progress)
4. Chunks → embedded into vector database (upcoming)
5. Query → retrieve relevant chunks → LLM answer (upcoming)

------------------------------------------------------------------------------------------------------------------------------
## Progress Log
------------------------------------------------------------------------------------------------------------------------------
- Day 1: Keyword extraction via Gemini working
- Day 1: arXiv paper downloader working
- Day 1: Git setup, .env secured

------------------------------------------------------------------------------------------------------------------------------
## Setup
------------------------------------------------------------------------------------------------------------------------------
pip install -r requirements.txt
Add your GEMINI_API_KEY to .env