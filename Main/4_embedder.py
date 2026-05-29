from sentence_transformers import SentenceTransformer
import chromadb
import json
import os

def load_chunks(chunks_path="data/papers/chunks.json"):
    with open(chunks_path, "r") as f:
        return json.load(f)

def build_vector_store(chunks, model, client):
    # Delete collection if exists to rebuild fresh
    try:
        client.delete_collection("papers")
    except:
        pass
    
    collection = client.create_collection("papers")

    print(f"Embedding {len(chunks)} chunks...")

    # Embed in batches of 50
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [{
            "arxiv_id": c["arxiv_id"],
            "title": c["title"],
            "chunk_index": c["chunk_index"]
        } for c in batch]

        embeddings = model.encode(texts).tolist()

        collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        print(f"  Embedded chunks {i+1} to {min(i+batch_size, len(chunks))}")

    print(f"\nVector store built. {collection.count()} chunks indexed.")
    return collection

def query_vector_store(query, model, collection, n_results=3):
    # Model is now passed as an argument so we don't reload it every time
    query_embedding = model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results

if __name__ == "__main__":
    print("Loading embedding model and database (this takes a moment)...")
    # 1. Load heavy assets exactly ONCE
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="data/vectorstore")
    
    # 2. Smart Build Logic: Only embed if the database is missing or outdated
    try:
        collection = client.get_collection("papers")
        chunks = load_chunks()
        if collection.count() != len(chunks):
            print("Chunk count mismatch detected. Rebuilding vector store...")
            collection = build_vector_store(chunks, model, client)
        else:
            print(f"✅ Found existing vector store with {collection.count()} chunks. Skipping rebuild.")
    except Exception:
        print("No existing vector store found. Building from scratch...")
        chunks = load_chunks()
        collection = build_vector_store(chunks, model, client)

    # 3. The Interactive Search Loop
    print("\n" + "="*60)
    print("🧠 PaperMind Semantic Search Online")
    print("="*60)

    while True:
        user_query = input("\nEnter your question (or type 'exit' to quit): ").strip()
        
        if user_query.lower() in ['exit', 'quit']:
            print("Shutting down search. Goodbye!")
            break
            
        if not user_query:
            continue

        print(f"\nSearching database for: '{user_query}'...")
        
        results = query_vector_store(user_query, model, collection, n_results=3)

        print("\n--- Top Results ---")
        for i, (doc, meta) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0]
        )):
            print(f"\n{i+1}. {meta['title'][:80]}...")
            print(f"   Chunk: {meta['chunk_index']} | ID: {meta['arxiv_id']}")
            print(f"   Text preview: {doc[:300]}...")
            print("-" * 60)