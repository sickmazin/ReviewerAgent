import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
import os

from dataset.config import OUTPUT_DIR, REPO_ID, TOTAL_LIMIT, OUTPUT_FILE

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_valid_review_files():
    """
    Queries the Hugging Face API to find actual files in the repository.
    """
    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    # Filter only files in raw review categories
    review_files = [f for f in files if f.startswith("raw/review_categories/") and (f.endswith(".jsonl") or f.endswith(".jsonl.gz"))]
    return review_files

def download_reviews():
    review_files = get_valid_review_files()
    if not review_files:
        print("Error: Could not find review files in the repository.")
        return

    # Select main categories for diversity
    target_files = [f for f in review_files if any(cat in f for cat in ["Books", "Electronics", "Home", "Clothing", "Toys", "Movies", "Pet", "Sports", "Beauty", "Health"])]
    
    limit_per_file = TOTAL_LIMIT // len(target_files)
    total_count = 0
    first_write = True

    print(f"Found {len(target_files)} category files. Downloading approximately {limit_per_file} rows for each.")

    for file_path in target_files:
        cat_name = file_path.split("/")[-1].split(".")[0]
        print(f"\nProcessing: {cat_name}...")
        
        # Direct URL for JSON loading
        file_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{file_path}"
        
        try:
            # Use 'json' loader which doesn't require .py scripts
            dataset = load_dataset("json", data_files=file_url, split="train", streaming=True)
            
            chunk = []
            file_count = 0
            pbar = tqdm(total=limit_per_file, desc=f"Streaming {cat_name}")
            
            for entry in dataset:
                if file_count >= limit_per_file:
                    break
                
                # Data extraction
                chunk.append({
                    'text': entry.get('text', ''),
                    'helpful_vote': entry.get('helpful_vote', 0),
                    'rating': entry.get('rating', 0),
                    'category': cat_name
                })
                
                file_count += 1
                total_count += 1
                pbar.update(1)
                
                # Incremental save
                if len(chunk) >= 50_000:
                    pd.DataFrame(chunk).to_csv(OUTPUT_FILE, mode='a', index=False, header=first_write, encoding='utf-8')
                    first_write = False
                    chunk = []
            
            if chunk:
                pd.DataFrame(chunk).to_csv(OUTPUT_FILE, mode='a', index=False, header=first_write, encoding='utf-8')
                first_write = False
            
            pbar.close()
            
        except Exception as e:
            print(f"Skipping {cat_name} due to error: {e}")
            continue

    print(f"\nFinished! Downloaded {total_count} reviews into {OUTPUT_FILE}")

if __name__ == "__main__":
    download_reviews()
