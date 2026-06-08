import os

import dotenv
from google import genai
from typing import Optional
import re
# Setup Gemini API
api_key = dotenv.get_key("../../.env", "GOOGLE_API_KEY")

client = None
if api_key:
    client = genai.Client(api_key=api_key)
def get_gemini_score(text: str, max_retries: int = 2) -> Optional[float]:
    """
    Invocazione di Gemini per ottenere un punteggio di insightfulness (0-100).
    Restituisce solo il numero.
    """
    if not client:
        print("[ERROR] Client Gemini non inizializzato. Verifica GOOGLE_API_KEY.")
        return None

    # Carica il prompt dal file dedicato
    prompt_file_path = os.path.join(os.path.dirname(__file__), "../", ".prompts", "benchmark.txt")
    try:
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        prompt = prompt_template.format(text=text)
    except Exception as e:
        print(f"[ERROR] Impossibile caricare il prompt: {e}")
        # Fallback prompt se il file non esiste
        prompt = f"Valuta l'insightfulness (0-100) di questa recensione, rispondi solo col numero: {text}"

    import time
    for attempt in range(max_retries + 1):
        try:
            model_name = 'gemini-3.1-flash-lite'

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            # Pulizia dell'output per estrarre solo il numero
            score_text = response.text.strip()
            match = re.search(r"(\d+(\.\d+)?)", score_text)
            if match:
                return float(match.group(1))
            return None
        except Exception as e:
            error_str = str(e)
            if attempt < max_retries:
                print(f"[WARNING] Errore Gemini ({model_name}): {error_str[:100]}... Ritento ({attempt + 1}/{max_retries}) tra 5s")
                time.sleep(5)
            else:
                print(f"[ERROR] Errore finale nella chiamata a Gemini: {e}")
                return None

if __name__ == "__main__":
    test_review = "Questo prodotto è fantastico, lo consiglio a tutti per la sua qualità costruttiva."
    score = get_gemini_score(test_review)
    print(f"Test Score: {score}")
