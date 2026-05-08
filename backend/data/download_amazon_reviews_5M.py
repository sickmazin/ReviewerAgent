import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
import os

# --- CONFIGURAZIONE ---
REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
TOTAL_LIMIT = 5_000_000
OUTPUT_DIR = "../../.datasets"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "amazon_reviews_5M_random.csv")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_valid_review_files():
    """
    Interroga l'API di Hugging Face per trovare i file reali nel repository.
    """
    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    # Filtriamo solo i file nelle categorie di review raw
    review_files = [f for f in files if f.startswith("raw/review_categories/") and (f.endswith(".jsonl") or f.endswith(".jsonl.gz"))]
    return review_files

def download_5m_reviews():
    review_files = get_valid_review_files()
    if not review_files:
        print("Errore: Impossibile trovare file di review nel repository.")
        return

    # Selezioniamo le categorie principali per diversità
    target_files = [f for f in review_files if any(cat in f for cat in ["Books", "Electronics", "Home", "Clothing", "Toys", "Movies", "Pet", "Sports", "Beauty", "Health"])]
    
    limit_per_file = TOTAL_LIMIT // len(target_files)
    total_count = 0
    first_write = True

    print(f"Trovati {len(target_files)} file di categoria. Scaricando circa {limit_per_file} righe per ognuno.")

    for file_path in target_files:
        cat_name = file_path.split("/")[-1].split(".")[0]
        print(f"\nProcessing: {cat_name}...")
        
        # URL diretto per il caricamento JSON
        file_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{file_path}"
        
        try:
            # Usiamo il loader 'json' che non richiede script .py
            dataset = load_dataset("json", data_files=file_url, split="train", streaming=True)
            
            chunk = []
            file_count = 0
            pbar = tqdm(total=limit_per_file, desc=f"Streaming {cat_name}")
            
            for entry in dataset:
                if file_count >= limit_per_file:
                    break
                
                # Estrazione dati
                chunk.append({
                    'text': entry.get('text', ''),
                    'helpful_vote': entry.get('helpful_vote', 0),
                    'rating': entry.get('rating', 0),
                    'category': cat_name
                })
                
                file_count += 1
                total_count += 1
                pbar.update(1)
                
                # Salvataggio incrementale
                if len(chunk) >= 50_000:
                    pd.DataFrame(chunk).to_csv(OUTPUT_FILE, mode='a', index=False, header=first_write, encoding='utf-8')
                    first_write = False
                    chunk = []
            
            if chunk:
                pd.DataFrame(chunk).to_csv(OUTPUT_FILE, mode='a', index=False, header=first_write, encoding='utf-8')
                first_write = False
            
            pbar.close()
            
        except Exception as e:
            print(f"Salto {cat_name} causa errore: {e}")
            continue

    print(f"\nFine! Scaricate {total_count} recensioni in {OUTPUT_FILE}")

if __name__ == "__main__":
    download_5m_reviews()
