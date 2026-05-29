from sentence_transformers import SentenceTransformer
import chromadb
import json
import os

def load_chunks(chunks_path="data/papers/chunks.json"):
    with open(chunks_path, "r") as f:
        return json.load(f)

def build_vector_store(chunks):
    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Setup ChromaDB
    client = chromadb.PersistentClient(path="data/vectorstore")
    
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

def query_vector_store(query, n_results=5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="data/vectorstore")
    collection = client.get_collection("papers")

    query_embedding = model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results

if __name__ == "__main__":
    chunks = load_chunks()
    collection = build_vector_store(chunks)

    # Test query
    print("\n--- Testing retrieval ---")
    test_query = "formation energy prediction graphene doping"
    results = query_vector_store(test_query)

    print(f"\nQuery: {test_query}")
    print("\nTop 3 results:")
    for i, (doc, meta) in enumerate(zip(
        results["documents"][0][:3],
        results["metadatas"][0][:3]
    )):
        print(f"\n{i+1}. {meta['title'][:60]}")
        print(f"   Chunk: {meta['chunk_index']}")
        print(f"   Text: {doc[:150]}...")