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
Your task is to understand the user research work and create keywords from your understanding.
Generate atleast 25 keywords for the creation of the list.
generates the keywords in such a way that those keywords will be used for doanloading the relevant research papers and would help to get the best downloading research papers. 
Output ONLY a Python list of keyword strings. Nothing else.
No explanation, no headings, no extra text.

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
        user_input = input("\nDescribe your research: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        elif user_input.strip():
            print("Extracting keywords...")
            raw_keywords = get_gemini_response(user_input)
            keywords = parse_keywords(raw_keywords)
            print(f"\nKeywords extracted: {len(keywords)}")
            print(keywords)

            # Step 2: Let user review and add keywords
            print("\nReview your keywords above.")
            user_addition = input("Add more keywords (comma separated) or press ENTER to skip: ").strip()

            if user_addition:
                extra = [k.strip() for k in user_addition.split(",") if k.strip()]
                keywords.extend(extra)
                print(f"\nUpdated list ({len(keywords)} keywords):")
                print(keywords)

            # Step 3: Confirm and download
            confirm = input("\nDownload papers for these keywords? (y/n): ")
            if confirm.lower() == "y":
                papers = download_papers(keywords, max_per_keyword=3)
                print(f"\nPipeline complete. {len(papers)} papers ready.")
            else:
                print("Download cancelled.")

        else:
            print("Please enter your research context.")