from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ System instruction — tells LLM exactly what to do
SYSTEM_PROMPT = """
You are a research keyword extractor for an academic paper scraper.
your task is understand the user research work, think about it detailed ass the research assistant, your ONLY 
job is take context and help of your idea around the this research topic create the keywords from your understanding 
atleast 25 keywords for the creation of the list.

Output ONLY a Python list of keyword strings. Nothing else.
No explanation, no headings, no extra text.

Example output:
["graphene doping machine learning", "formation energy DFT graphene", "SOAP descriptor materials"]
"""

def get_gemini_response(user_prompt):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{SYSTEM_PROMPT}\n\nUser research context:\n{user_prompt}"
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    print("--- PaperMInd Keyword Extractor ---")
    
    while True:
        user_input = input("\nDescribe your research: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        elif user_input.strip():
            print("Extracting keywords...")
            keywords = get_gemini_response(user_input)
            print(f"\nKeywords:\n{keywords}")
        else:
            print("Please enter your research context.")