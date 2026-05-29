import arxiv
import os
import json
import requests
import time
import random

RATE_LIMIT_SECONDS = 5

def build_combined_query(keywords, batch_size=3):
    batches = []
    for i in range(0, len(keywords), batch_size):
        batch = keywords[i:i + batch_size]
        combined = " OR ".join(kw for kw in batch)
        batches.append(combined)
    return batches

def fetch_with_backoff(client, search, retries=4):
    for attempt in range(retries):
        try:
            results = list(client.results(search))
            return results
        except Exception as e:
            if "429" in str(e) or "503" in str(e):
                wait = RATE_LIMIT_SECONDS * (2 ** attempt) + random.uniform(1, 3)
                print(f"  Rate limited. Waiting {wait:.1f}s before retry {attempt+1}...")
                time.sleep(wait)
            else:
                print(f"  Error: {e}")
                return []
    print("  Failed after all retries.")
    return []

def is_relevant(result, keywords):
    text = (result.title + " " + result.summary).lower()
    for kw in keywords:
        if any(word.lower() in text for word in kw.split()):
            return True
    return False

def download_papers(keywords, max_per_keyword=2, save_dir="data/papers"):
    os.makedirs(save_dir, exist_ok=True)

    all_papers = []
    seen_ids = set()

    # Load existing metadata to avoid re-downloading
    metadata_path = os.path.join(save_dir, "papers_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            existing = json.load(f)
            for p in existing:
                seen_ids.add(p.get("arxiv_id", ""))
        print(f"Found {len(seen_ids)} already downloaded papers. Skipping duplicates.")

    # Batch keywords
    query_batches = build_combined_query(keywords, batch_size=3)
    print(f"\nSearching {len(keywords)} keywords in {len(query_batches)} batched queries.\n")

    for i, query in enumerate(query_batches):
        print(f"Batch {i+1}/{len(query_batches)}: {query[:80]}...")

        time.sleep(RATE_LIMIT_SECONDS + random.uniform(1, 2))

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_per_keyword * 3,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = fetch_with_backoff(client, search)
        count = 0

        for result in results:
            arxiv_id = result.entry_id.split("/")[-1]

            if arxiv_id in seen_ids:
                continue

            if not is_relevant(result, keywords):
                continue

            seen_ids.add(arxiv_id)

            paper = {
                "arxiv_id": arxiv_id,
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "summary": result.summary,
                "pdf_url": result.pdf_url,
                "published": str(result.published),
                "categories": list(result.categories),
                "keyword_source": query,
                "local_path": None
            }

            filename = arxiv_id + ".pdf"
            filepath = os.path.join(save_dir, filename)

            if os.path.exists(filepath):
                print(f"  Already exists: {result.title[:60]}")
                paper["local_path"] = filepath
            else:
                try:
                    time.sleep(RATE_LIMIT_SECONDS)
                    response = requests.get(result.pdf_url, timeout=30)
                    response.raise_for_status()
                    with open(filepath, "wb") as f:
                        f.write(response.content)
                    paper["local_path"] = filepath
                    print(f"  Downloaded: {result.title[:60]}")
                except Exception as e:
                    print(f"  Download failed: {e}")

            all_papers.append(paper)
            count += 1

            if count >= max_per_keyword:
                break

    # Save metadata
    with open(metadata_path, "w") as f:
        json.dump(all_papers, f, indent=2)

    print(f"\nTotal papers downloaded: {len(all_papers)}")
    return all_papers