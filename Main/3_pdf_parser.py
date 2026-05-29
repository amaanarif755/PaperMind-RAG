import pymupdf4llm
import fitz
import os
import json
import re

def extract_text_markdown(filepath):
    """
    Extract text as Markdown using pymupdf4llm.
    Better than raw fitz for scientific papers — preserves tables and headers.
    """
    try:
        md_text = pymupdf4llm.to_markdown(filepath)
        return md_text
    except Exception as e:
        print(f"  pymupdf4llm failed: {e}, falling back to fitz...")
        return extract_text_fallback(filepath)

def extract_text_fallback(filepath):
    """
    Fallback: basic fitz extraction if pymupdf4llm fails.
    """
    doc = fitz.open(filepath)
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text.strip())
    doc.close()
    return "\n".join(pages)

def clean_text(text):
    """
    Remove noise that hurts RAG retrieval:
    - References section
    - Excessive whitespace
    - Page numbers
    - URLs (keep DOIs though)
    """
    # Remove references section (everything after References/Bibliography)
    text = re.split(r'\n#+\s*(References|Bibliography|REFERENCES)\s*\n', text)[0]

    # Remove lines that are just page numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove URLs (not DOIs)
    text = re.sub(r'http[s]?://(?!doi)\S+', '', text)

    return text.strip()

def chunk_by_section(text, chunk_size=1000, overlap=100):
    """
    Split text into chunks.
    Strategy: split by markdown headers first (sections),
    then split large sections by character count with overlap.
    Injects section headers into continuing sub-chunks to preserve context.
    """
    chunks = []

    # Split by markdown headers
    sections = re.split(r'\n(?=#{1,3} )', text)

    for section in sections:
        if not section.strip():
            continue

        # --- EXTRACT HEADER CONTEXT ---
        # Capture the markdown heading (e.g., "## 3. Methodology") to inject into child chunks
        header_match = re.match(r'^(#{1,3}\s[^\n]+)', section)
        header_context = header_match.group(1) + "\n[Context Continued]... " if header_match else ""

        if len(section) <= chunk_size:
            chunks.append(section.strip())
        else:
            words = section.split()
            current_chunk = []
            # Account for the header length inside the chunk size target
            current_length = len(header_context)

            for word in words:
                current_chunk.append(word)
                current_length += len(word) + 1

                if current_length >= chunk_size:
                    # Prepend structural context before appending chunk text
                    chunk_text = header_context + " ".join(current_chunk)
                    chunks.append(chunk_text.strip())
                    
                    # Sliding window overlap calculation
                    overlap_words = current_chunk[-max(1, overlap//5):]
                    current_chunk = overlap_words
                    current_length = len(header_context) + sum(len(w) + 1 for w in overlap_words)

            if current_chunk and len(current_chunk) > (overlap//5):
                chunk_text = header_context + " ".join(current_chunk)
                chunks.append(chunk_text.strip())

    return chunks

def parse_all_papers(save_dir="data/papers"):
    metadata_path = os.path.join(save_dir, "papers_metadata.json")

    if not os.path.exists(metadata_path):
        print(f"❌ Metadata file not found at {metadata_path}. Please run Stage 1 & 2 first.")
        return []

    with open(metadata_path, "r") as f:
        papers = json.load(f)

    all_chunks = []

    for paper in papers:
        if not paper.get("local_path"):
            continue
        if not os.path.exists(paper["local_path"]):
            continue

        print(f"\nParsing: {paper['title'][:60]}...")

        # Extract
        raw_text = extract_text_markdown(paper["local_path"])

        # Clean
        clean = clean_text(raw_text)

        # Chunk
        chunks = chunk_by_section(clean)

        print(f"  Characters: {len(clean)} | Chunks: {len(chunks)}")

        # Store chunks with metadata
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "chunk_id": f"{paper['arxiv_id']}_chunk_{i}",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "text": chunk
            })

    # Save chunks
    chunks_path = os.path.join(save_dir, "chunks.json")
    with open(chunks_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nTotal chunks created: {len(all_chunks)}")
    print(f"Saved to: {chunks_path}")
    return all_chunks

if __name__ == "__main__":
    chunks = parse_all_papers()

    # Preview first chunk
    if chunks:
        print("\n--- PREVIEW: First Chunk ---")
        print(f"Paper: {chunks[0]['title'][:60]}")
        print(f"Chunk ID: {chunks[0]['chunk_id']}")
        print(f"Text preview:\n{chunks[0]['text'][:300]}")