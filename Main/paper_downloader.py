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
        # Group each multi-word phrase in parentheses to protect parser logic
        combined = " OR ".join(f"({kw})" for kw in batch)
        batches.append(combined)
    return batches

def fetch_with_backoff(client, search, retries=5):
    base_delay = 10  # Heavy start delay for recovery
    for attempt in range(retries):
        try:
            # Force generator evaluation inside the try-except block
            return list(client.results(search))
        except arxiv.HTTPError as e:
            if e.status in [429, 503]:
                wait = base_delay * (2 ** attempt) + random.uniform(2, 5)
                print(f"  ⚠️ arXiv Rate Limited ({e.status}). Cooling down for {wait:.1f}s (Attempt {attempt+1}/{retries})...")
                time.sleep(wait)
            else:
                print(f"  ❌ arXiv HTTP Error: {e}")
                return []
        except Exception as e:
            print(f"  ❌ Unexpected Error: {e}")
            return []
    print("  ❌ Failed to fetch after all retries due to rate limits.")
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

    # FIX 1: Load existing metadata into memory so we APPEND, not OVERWRITE
    metadata_path = os.path.join(save_dir, "papers_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            try:
                all_papers = json.load(f)
                for p in all_papers:
                    seen_ids.add(p.get("arxiv_id", ""))
            except json.JSONDecodeError:
                print("  ⚠️ Metadata file corrupted. Starting fresh.")
                all_papers = []
        print(f"Found {len(seen_ids)} already downloaded papers. Keeping them in metadata history.")

    query_batches = build_combined_query(keywords, batch_size=3)
    print(f"\nSearching {len(keywords)} keywords in {len(query_batches)} batched queries.\n")

    # FIX 2: Instantiate client ONCE outside the loop with safe delays
    client = arxiv.Client(
        page_size=15,
        delay_seconds=4.5,
        num_retries=4
    )

    for i, query in enumerate(query_batches):
        print(f"Batch {i+1}/{len(query_batches)}: {query[:90]}...")

        # Generous breathing room between batch requests
        time.sleep(RATE_LIMIT_SECONDS + random.uniform(1, 3))

        search = arxiv.Search(
            query=query,
            max_results=max_per_keyword * 4,
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

            filename = f"{arxiv_id}.pdf"
            filepath = os.path.join(save_dir, filename)

            if os.path.exists(filepath):
                print(f"  -> File exists locally: {result.title[:50]}...")
                paper["local_path"] = filepath
                all_papers.append(paper)
                count += 1
            else:
                try:
                    # Respectful sleep before hitting the PDF download endpoint
                    time.sleep(3.0 + random.uniform(0.5, 1.5))
                    response = requests.get(result.pdf_url, timeout=30)
                    response.raise_for_status()
                    
                    with open(filepath, "wb") as f:
                        f.write(response.content)
                    
                    paper["local_path"] = filepath
                    all_papers.append(paper)
                    print(f"  📥 Downloaded: {result.title[:50]}...")
                    count += 1
                except Exception as e:
                    print(f"  ❌ Download failed for {arxiv_id}: {e}")

            if count >= max_per_keyword:
                break

    # Save the cumulative updated metadata
    with open(metadata_path, "w") as f:
        json.dump(all_papers, f, indent=2)

    print(f"\n✅ Pipeline Stage 2 Complete. Total tracked papers in database: {len(all_papers)}")
    return all_papers