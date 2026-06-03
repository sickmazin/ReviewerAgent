import os
import time
import pandas as pd
from typing import List, Optional
import sys
import re
import dotenv

# Aggiungi la root del progetto al path per gli import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.ml_models.Insightfulness_Model import Insightfulness
from backend.comparison.gemini import get_gemini_score
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

def get_ollama_score(text: str, model_name: str = "gpt-oss:120b-cloud", max_retries: int = 2) -> Optional[float]:
    """
    Invocazione di Ollama per ottenere un punteggio di insightfulness (0-100).
    Restituisce solo il numero.
    """
    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
    ollama_base_url = os.getenv("OLLAMA_API_URL")
    ollama_api_key = os.getenv("OLLAMA_API_KEY")
    
    headers = {}
    if ollama_api_key:
        auth_value = ollama_api_key if ollama_api_key.startswith("Bearer ") else f"Bearer {ollama_api_key}"
        headers["Authorization"] = auth_value
        headers["X-API-Key"] = ollama_api_key.replace("Bearer ", "")

    for attempt in range(max_retries + 1):
        try:
            llm = ChatOllama(
                model=model_name,
                temperature=0.0,
                base_url=ollama_base_url,
                headers=headers if headers else None
            )
            
            prompt_file_path = os.path.join(os.path.dirname(__file__), "../", ".prompts", "benchmark.txt")
            try:
                with open(prompt_file_path, "r", encoding="utf-8") as f:
                    prompt_template = f.read()
            except Exception as e:
                print(f"[ERROR] Impossibile caricare il prompt per Ollama: {e}")
                prompt_template = "Valuta l'insightfulness (0-100) di questa recensione, rispondi solo col numero: {text}"
            
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | llm
            
            raw_response = chain.invoke({"text": text})
            content = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
            
            score_text = content.strip()
            match = re.search(r"(\d+(\.\d+)?)", score_text)
            if match:
                return float(match.group(1))
            return None
            
        except Exception as e:
            error_str = str(e)
            if attempt < max_retries:
                print(f"[WARNING] Errore Ollama ({model_name}): {error_str[:100]}... Ritento ({attempt + 1}/{max_retries}) tra 5s")
                time.sleep(5)
            else:
                print(f"[ERROR] Errore finale Ollama ({model_name}) dopo {max_retries} tentativi: {e}")
                return None

def run_benchmark(dataset_path: str, num_reviews: int = 50):
    """
    Esegue il benchmark comparativo tra il modello locale Insightfulness, Gemini e Ollama.
    """
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset non trovato a: {dataset_path}")
        return

    print(f"Caricamento dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Assumiamo che la colonna con il testo si chiami 'text'
    if 'text' not in df.columns:
        # Prova a cercare colonne comuni
        potential_cols = ['review', 'review_body', 'content', 'text']
        for col in potential_cols:
            if col in df.columns:
                df = df.rename(columns={col: 'text'})
                break
        else:
            print("[ERROR] Colonna di testo non trovata nel dataset.")
            return

    # Prendi le prime 50 recensioni valide
    reviews = df['text'].dropna().astype(str).tolist()[:num_reviews]
    
    print(f"Inizializzazione modello Insightfulness...")
    # Percorso pesi (adattalo se necessario, qui usiamo quello standard di Insightfulness_Model.py)
    MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.weights/v7/epoch_117.pt"))
    if not os.path.exists(MODEL_PATH):
        # Fallback per il test main se i pesi sono altrove
        MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.weights/v7_frozen/best.pt"))

    insight_model = Insightfulness(model_path=MODEL_PATH)
    
    results = []

    print(f"Inizio benchmark su {len(reviews)} recensioni...")
    
    for i, review in enumerate(reviews):
        print(f"[{i+1}/{len(reviews)}] Elaborazione...")
        
        # Benchmark Insightfulness
        start_time = time.time()
        try:
            # Chiamiamo direttamente la pipeline del scorer per avere il numero (0-100)
            score_insight = insight_model.scorer.inference_pipeline(review, insight_model._tokenizer)
        except Exception as e:
            print(f"Errore Insightfulness: {e}")
            score_insight = None
        time_insight = time.time() - start_time
        
        # Benchmark Gemini
        start_time = time.time()
        score_gemini = get_gemini_score(review)
        time_gemini = time.time() - start_time
        
        # Benchmark Ollama Cloud
        start_time = time.time()
        score_ollama = get_ollama_score(review, model_name="gpt-oss:120b-cloud")
        time_ollama = time.time() - start_time
        
        results.append({
            'review_id': i,
            'review_text': review[:100] + "...", # Salviamo solo un'anteprima
            'score_insight': score_insight,
            'time_insight': time_insight,
            'score_gemini': score_gemini,
            'time_gemini': time_gemini,
            'score_ollama': score_ollama,
            'time_ollama': time_ollama
        })

    # Salva i risultati
    output_df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(__file__), "benchmark_results.csv")
    output_df.to_csv(output_path, index=False)
    
    print(f"\nBenchmark completato! Risultati salvati in: {output_path}")
    
    # Riassunto
    print("\nRIASSUNTO TEMPI MEDI:")
    print(f"Insightfulness: {output_df['time_insight'].mean():.4f}s")
    print(f"Gemini:         {output_df['time_gemini'].mean():.4f}s")
    print(f"Ollama Cloud:   {output_df['time_ollama'].mean():.4f}s")

if __name__ == "__main__":
    
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.datasets/reviews_labeled.csv"))
    
    run_benchmark(dataset_path)
