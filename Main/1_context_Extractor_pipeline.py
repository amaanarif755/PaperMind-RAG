from google import genai
from dotenv import load_dotenv
import os
import time
import ast

from paper_downloader import download_papers

def parse_keywords(raw_text):
    try:
        keywords = ast.literal_eval(raw_text.strip())
        return keywords
    except:
        return [k.strip().strip('"') for k in raw_text.strip("[]").split(",")]

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """
You are a research keyword extractor for an academic paper scraper.
Understand the user's research topic deeply and generate keywords a researcher
would actually search for on arXiv.

Rules:
- Generate at least 25 keywords relevent for the search of academic papers on arXiv
- Keywords should be 2-5 words each
- Focus on technical terms, not generic phrases
- Output ONLY a Python list of strings. Nothing else.
- No explanation, no headings, no extra text.

Example output:
["graphene doping machine learning", "formation energy DFT graphene", "SOAP descriptor materials"]
"""

def get_gemini_response(user_prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{SYSTEM_PROMPT}\n\nUser research context:\n{user_prompt}"
            )
            return response.text
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
    return "Error: All retries failed"

if __name__ == "__main__":
    print("--- PaperMind Keyword Extractor ---")

    while True:
        user_input = input("\nDescribe your research (or type 'exit'): ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        elif user_input.strip():
            print("\nExtracting keywords...")
            raw_keywords = get_gemini_response(user_input)
            keywords = parse_keywords(raw_keywords)
            print(f"\nKeywords extracted: {len(keywords)}")
            for i, kw in enumerate(keywords, 1):
                print(f"  {i}. {kw}")

            # Let user add keywords
            print("\nReview keywords above.")
            user_addition = input("Add more (comma separated) or press ENTER to skip: ").strip()

            if user_addition and user_addition.lower() != "enter":
                extra = [k.strip() for k in user_addition.split(",") if k.strip()]
                keywords.extend(extra)
                print(f"Updated: {len(keywords)} total keywords.")

            # Confirm and download
            confirm = input("\nDownload papers? (y/n): ")
            if confirm.lower() == "y":
                # Use first 15 keywords only to avoid rate limits
                search_keywords = keywords[:15]
                print(f"\nSearching with top {len(search_keywords)} keywords...")
                papers = download_papers(search_keywords, max_per_keyword=2)
                print(f"\nPipeline complete. {len(papers)} papers ready for parsing.")
            else:
                print("Download cancelled.")

        else:
            print("Please enter your research context.")