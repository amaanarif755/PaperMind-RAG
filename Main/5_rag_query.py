from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from dotenv import load_dotenv
import os
import time

load_dotenv()
# Initialize the modern Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def retrieve_chunks(query, model, collection, n_results=5):
    """Retrieves relevant text chunks using pre-loaded model and collection objects."""
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results

def build_context(results):
    """Formats the retrieved database results into a structured string for the LLM."""
    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(
            f"[Paper: {meta['title'][:60]} | Chunk: {meta['chunk_index']}]\n{doc}\n"
        )
    return "\n---\n".join(context_parts)

def ask_gemini(query, context, retries=3):
    """Sends the context and question to Gemini with resilient retry logic."""
    system_instruction = (
        "You are a meticulous research assistant helping a researcher understand scientific papers.\n"
        "Answer the question using ONLY the context provided. If the answer is not contained within "
        "the context, explicitly state: 'I could not find this in the downloaded papers.'\n"
        "Be highly specific, professional, and explicitly cite which paper and chunk your facts come from."
    )

    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config={"system_instruction": system_instruction}
            )
            return response.text
        except Exception as e:
            print(f"⚠️ Gemini API Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                print("  Retrying in 5 seconds...")
                time.sleep(5)
            else:
                return f"Error: API calls failed after {retries} attempts. Details: {e}"

def rag_query(question, model, collection):
    """Orchestrates the entire Retrieval-Augmented Generation workflow."""
    print(f"\nSearching relevant chunks...")
    results = retrieve_chunks(question, model, collection, n_results=5)

    print(f"Retrieved {len(results['documents'][0])} chunks from:")
    seen = set()
    for meta in results["metadatas"][0]:
        if meta["title"] not in seen:
            print(f"  - {meta['title'][:60]}")
            seen.add(meta["title"])

    context = build_context(results)
    print("\nGenerating answer...")
    answer = ask_gemini(question, context)
    return answer

if __name__ == "__main__":
    print("=== PaperMind RAG System ===")
    print("Loading local embeddings engine and database...")
    
    # Load heavy framework assets exactly ONCE at startup
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="data/vectorstore")
    
    try:
        paper_collection = chroma_client.get_collection("papers")
    except Exception as e:
        print(f"❌ Could not load vector collection: {e}. Please run Main/4_embedder.py first.")
        exit(1)

    print("\nPaperMind RAG Engine Online.")
    print("Ask questions about your downloaded research papers.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not question:
            continue

        # Pass the initialized models straight down into the workflow loop
        answer = rag_query(question, embedding_model, paper_collection)
        print(f"\nAnswer:\n{answer}")
        print("\n" + "="*60)