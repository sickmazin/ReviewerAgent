import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, DebertaV2Config
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import login
import os


# Caricamento modelli linguistici
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

load_dotenv()
token = os.getenv("HUGGING_FACE_TOKEN")
if token:
    login(token=token)


class InsightScorer(nn.Module):
    """
    Architettura Multi-Task per Insightfulness Scoring.
    Metodi inclusi: Distillazione LLM (Regression), Subjectivity Analysis, Aspect Detection.
    """
    def __init__(self, model_name="microsoft/deberta-v3-small", n_aspects=10):
        super(InsightScorer, self).__init__()
        self.config = DebertaV2Config.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Teste del Modello (Metodo 5: Multi-task)
        self.regressor = nn.Linear(self.config.hidden_size, 1)        # Score Primario (0-100)
        self.subjectivity_head = nn.Linear(self.config.hidden_size, 1) # Obiettività (0: Fatto, 1: Opinione)
        self.aspect_head = nn.Linear(self.config.hidden_size, n_aspects) # Copertura Attributi

        # Sincronizzazione Dtype: Forziamo tutto in float32 per evitare mismatch con Half/float16
        self.float()

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        # Utilizziamo il CLS token (o pooler output) per la rappresentazione semantica globale
        cls_emb = outputs.last_hidden_state[:, 0, :]

        score = torch.sigmoid(self.regressor(cls_emb)) * 100
        subj = torch.sigmoid(self.subjectivity_head(cls_emb))
        aspects = torch.sigmoid(self.aspect_head(cls_emb))

        return score, subj, aspects, cls_emb

class UnsupervisedEngine:
    """
    Calcolo delle metriche euristiche e informative (Metodi 3 e 4).
    """
    def __init__(self, category_centroid_emb=None):
        self.category_centroid = category_centroid_emb # Embedding medio della categoria

    def get_lexical_density(self, text):
        doc = nlp(text)
        # Metodo 3: Densità di Entità e Aspetti (PROPN, NOUN, ADJ)
        entities = len(doc.ents)
        aspects = len([token for token in doc if token.pos_ in ["NOUN", "ADJ"]])
        total_tokens = len(doc) + 1e-9
        return (entities + aspects) / total_tokens

    def get_information_gain(self, review_emb):
        # Metodo 4: Semantic Entropy (Distanza dal rumore di fondo della categoria)
        if self.category_centroid is None:
            return 0.5

        # Reshape per scikit-learn
        review_emb = review_emb.reshape(1, -1)
        centroid = self.category_centroid.reshape(1, -1)

        # 1 - CosSim: Più è diversa dalla media, più l'informazione è specifica (Potenziale Insight)
        dist = 1 - cosine_similarity(review_emb, centroid)[0][0]
        return dist

class HybridInsightSystem:
    def __init__(self, model_path=None, category_centroid=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
        self.model = InsightScorer().to(self.device)

        if model_path and os.path.exists(model_path):
            print(f"Loading fine-tuned model from {model_path}...")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        elif model_path:
            print(f"Warning: model_path {model_path} not found. Using default weights.")

        self.model.eval()
        self.engine = UnsupervisedEngine(category_centroid)

    def compute_score(self, text, helpful_votes_proxy=0):
        """
        Calcola lo score finale integrando i 5 segnali.
        S_final = (S_model * (1 - Subj)) + (S_votes * w1) + (S_density * w2) + (S_gain * w3)
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            s_model, subj, aspects, emb = self.model(inputs['input_ids'], inputs['attention_mask'])

        # Estrazione metriche unsupervised
        lex_density = self.engine.get_lexical_density(text)
        # Convertiamo il tensore in numpy per l'engine unsupervised
        emb_np = emb.detach().cpu().numpy()[0]
        info_gain = self.engine.get_information_gain(emb_np)

        # Pesi di calibrazione (Hyperparameters)
        w_subj = 1.0 - subj.item() # Più è oggettiva, più lo score pesa
        w_votes = np.log1p(helpful_votes_proxy) / 10 # Metodo 1: Normalizzazione logaritmica voti

        # Algoritmo di Unificazione:
        # 1. Base score dalla distillazione LLM (s_model) pesata per l'obiettività
        # 2. Bonus per densità lessicale (specificità tecnica)
        # 3. Bonus per information gain (unicità dell'informazione)
        # 4. Bias positivo per i voti della community

        final_score = (s_model.item() * w_subj * 1.5) + \
                      (lex_density * 0.8) + \
                      (info_gain * 0.1) + \
                      (w_votes * 0.55)

        return np.clip(final_score, 0, 100)

# Esempio di utilizzo e test rapido
if __name__ == "__main__":
    MODEL_PATH = "../../.weights/v1/insight_model_epoch_67.pth"

    # Mock category centroid (embedding medio di 768 dim per DeBERTa small)
    mock_centroid = np.random.randn(768)

    # Inizializzazione con il modello fine-tuned
    system = HybridInsightSystem(model_path=MODEL_PATH, category_centroid=mock_centroid)

    review_alta = '''
        Queste infradito Cressi Saint-Tropez sono state una bellissima scoperta, rivelandosi un prodotto di ottima fattura che unisce l'affidabilità del marchio a un design moderno e vivace. La prima cosa che colpisce è l'abbinamento cromatico, in particolare la tonalità azzurrina: è un colore fresco, prettamente estivo e luminoso che risalta tantissimo. Nonostante siano pensate per bambine e ragazze, la calzata è generosa e versatile, risultando perfetta anche per un piede adulto che oscilla tra il 36 e il 37, garantendo un appoggio naturale e stabile.
        Dal punto di vista tecnico, la qualità dei materiali è evidente. Il plantare in gomma è compatto e sostiene bene il piede senza cedere sul tallone, mentre il cinturino a Y è incastrato saldamente nella suola, trasmettendo una sensazione di robustezza superiore rispetto ai modelli economici. Sono incredibilmente leggere e occupano pochissimo spazio nella borsa, ma non sacrificano la sicurezza: la suola ruvida offre un ottimo grip antiscivolo, fondamentale per muoversi senza timore su superfici bagnate o negli spogliatoi della palestra.
        Il comfort è assoluto anche dopo molte ore di utilizzo; il separatore tra le dita è morbido e sottile, il che permette di usarle per lunghe passeggiate sulla spiaggia senza il rischio di fastidiose vesciche. Essendo realizzate con materiali resistenti all'acqua, si asciugano in un lampo e sono semplicissime da pulire. Anche se il prezzo può sembrare leggermente più alto della media, la durata nel tempo e la cura dei dettagli giustificano pienamente l'investimento. Sono diventate le compagne inseparabili per il mare e la doccia, promosse a pieni voti per praticità e stile.
    '''
    review_bassa = "Belle, semplice e spedizione veloce."

    score_high = system.compute_score(review_alta, helpful_votes_proxy=0)
    score_low = system.compute_score(review_bassa, helpful_votes_proxy=0)

    print(f"\nModel: {'Fine-tuned' if os.path.exists(MODEL_PATH) else 'Default DeBERTa'}")
    print(f"Insight Score (High Quality): {score_high:.2f}/100")
    print(f"Insight Score (Low Quality): {score_low:.2f}/100")
