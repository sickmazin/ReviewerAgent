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

# Mappatura ID Categorie per Domain Embedding
CATEGORIES = [
    "electronics", "home_improvement", "fashion", "beauty", 
    "books", "sports", "automotive", "other"
]
cat_to_id = {c: i for i, c in enumerate(CATEGORIES)}

class SentenceAttention(nn.Module):
    """
    Meccanismo di attenzione per pesare l'importanza delle diverse frasi.
    """
    def __init__(self, hidden_dim):
        super(SentenceAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, sentence_embs, mask=None):
        # sentence_embs: [batch, n_sentences, hidden_dim]
        scores = self.attn(sentence_embs).squeeze(-1) # [batch, n_sentences]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = F.softmax(scores, dim=-1) # [batch, n_sentences]
        weighted_sum = torch.bmm(weights.unsqueeze(1), sentence_embs).squeeze(1)
        return weighted_sum, weights

class InsightScorer(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-small", n_aspects=10, domain_dim=32):
        super(InsightScorer, self).__init__()
        self.config = DebertaV2Config.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Domain Embedding Layer
        self.domain_emb = nn.Embedding(len(CATEGORIES), domain_dim)
        
        # Sentence Attention
        self.sentence_attention = SentenceAttention(self.config.hidden_size)
        
        # Regressore Finale (Concatenazione Doc Representation + Domain Embedding)
        input_dim = self.config.hidden_size + domain_dim
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.subjectivity_head = nn.Linear(input_dim, 1)
        self.aspect_head = nn.Linear(input_dim, n_aspects)
        
        self.float()
        
    def forward(self, input_ids, attention_mask, category_id):
        # In questa versione semplificata per efficienza, usiamo i token come proxy 
        # delle frasi applicando l'attenzione sugli stati nascosti.
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        token_embs = outputs.last_hidden_state # [batch, seq_len, hidden_size]
        
        # 1. Calcolo Rappresentazione Globale via Attention
        doc_emb, attn_weights = self.sentence_attention(token_embs, mask=attention_mask)
        
        # 2. Iniezione Domain Context
        d_emb = self.domain_emb(category_id) # [batch, domain_dim]
        combined = torch.cat([doc_emb, d_emb], dim=-1) # [batch, hidden+domain]
        
        # 3. Heads
        score = torch.sigmoid(self.regressor(combined)) * 100
        subj = torch.sigmoid(self.subjectivity_head(combined))
        aspects = torch.sigmoid(self.aspect_head(combined))
        
        return score, subj, aspects, doc_emb, attn_weights

class HybridInsightSystem:
    def __init__(self, model_path=None, category_centroid=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
        self.model = InsightScorer().to(self.device)
        
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
        self.model.eval()

    def compute_score(self, text, category="other"):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        cat_id = torch.tensor([cat_to_id.get(category, cat_to_id["other"])]).to(self.device)
        
        with torch.no_grad():
            s_model, subj, aspects, doc_emb, weights = self.model(
                inputs['input_ids'], 
                inputs['attention_mask'],
                cat_id
            )
        
        return s_model.item()

if __name__ == "__main__":
    system = HybridInsightSystem()
    test_text = "The battery life is amazing, lasted 15 hours of video playback."
    score = system.compute_score(test_text, category="electronics")
    print(f"Insight Score: {score:.2f}/100")
