from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from dotenv import load_dotenv
import os
import time

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def retrieve_chunks(query, n_results=5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="data/vectorstore")
    collection = chroma_client.get_collection("papers")

    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results

def build_context(results):
    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(
            f"[Paper: {meta['title'][:60]}]\n{doc}\n"
        )
    return "\n---\n".join(context_parts)

def ask_gemini(query, context, retries=3):
    prompt = f"""You are a research assistant helping a researcher understand scientific papers.

Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I could not find this in the downloaded papers."
Be specific and cite which paper your answer comes from.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
            return f"Error: {e}"

def rag_query(question):
    print(f"\nSearching relevant chunks...")
    results = retrieve_chunks(question, n_results=5)

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
    print("Ask questions about your downloaded research papers.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not question:
            continue

        answer = rag_query(question)
        print(f"\nAnswer:\n{answer}")
        print("\n" + "="*50)