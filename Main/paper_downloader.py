import arxiv
import os
import json
import requests

def download_papers(keywords, max_per_keyword=3, save_dir="data/papers"):
    os.makedirs(save_dir, exist_ok=True)
    
    all_papers = []
    seen_ids = set()
    
    for keyword in keywords:
        print(f"\nSearching: {keyword}")
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=keyword,
            max_results=max_per_keyword,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        for result in client.results(search):
            if result.entry_id in seen_ids:
                continue
            seen_ids.add(result.entry_id)
            
            paper = {
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "summary": result.summary,
                "pdf_url": result.pdf_url,
                "published": str(result.published),
                "keyword_source": keyword
            }
            
            # Download PDF using requests directly
            filename = result.entry_id.split("/")[-1] + ".pdf"
            filepath = os.path.join(save_dir, filename)
            
            try:
                response = requests.get(result.pdf_url, timeout=30)
                with open(filepath, "wb") as f:
                    f.write(response.content)
                paper["local_path"] = filepath
                print(f"  Downloaded: {result.title[:60]}")
            except Exception as e:
                print(f"  Failed: {e}")
                paper["local_path"] = None
            
            all_papers.append(paper)
    
    # Save metadata
    metadata_path = os.path.join(save_dir, "papers_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(all_papers, f, indent=2)
    
    print(f"\nTotal papers downloaded: {len(all_papers)}")
    return all_papers