"""
insight_model.py
================
InsightReviewScorer — modello unificato per lo scoring di insightfulness
di recensioni di prodotto.

Architettura
------------
  Encoder (DeBERTa-v3-small, distilbert, o qualsiasi HF AutoModel)
      |
      ├─ AttentionPooling        → rappresentazione globale pesata (migliore del solo CLS
      │                            per recensioni lunghe e strutturate)
      │
      ├─ score_head              → regressione 0-100 (distillazione LLM teacher)   [alpha]
      ├─ factuality_head         → densità informativa/fattualità offline           [beta]
      └─ lexical_head            → features spaCy fuse nello spazio latente         [delta]

Loss composita
--------------
  L = alpha * L_distill   (MSE score vs LLM teacher)
    + beta  * L_factuality (MSE fattualità predetta vs segnale offline)
    + gamma * L_bound      (allineamento geometrico worst→best per categoria)
    + delta * L_rank       (margin ranking: review_A > review_B se score_A > score_B)

Inferenza
---------
  Nessuna dipendenza da spaCy o da dati di categoria a runtime:
  l'encoder impara tutto ciò che serve dal testo grezzo.
  Le feature spaCy vengono usate solo come segnale di supervisione durante il training.
"""

import os
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# ============================================================
#  Attention Pooling
# ============================================================

class AttentionPooling(nn.Module):
    """
    Pooling pesato sull'intera sequenza di token tramite un attention scorer.
    Restituisce una rappresentazione (B, H) che cattura meglio le recensioni
    lunghe rispetto al solo token [CLS].
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(
            self,
            hidden_states: torch.Tensor,        # (B, T, H)
            attention_mask: torch.Tensor,       # (B, T)
    ) -> torch.Tensor:                      # (B, H)
        # Score di attenzione per ogni token
        scores = self.attn(hidden_states).squeeze(-1)           # (B, T)

        # Maschera padding con -inf prima della softmax
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)   # (B, T, 1)

        pooled = (hidden_states * weights).sum(dim=1)           # (B, H)
        return pooled


# ============================================================
#  Modello principale
# ============================================================

class InsightReviewScorer(nn.Module):
    """
    Scorer end-to-end per l'insightfulness di recensioni.

    Parametri
    ----------
    model_name      : nome HuggingFace del modello encoder (default: deberta-v3-large)
    n_lexical_feats : numero di feature lessicali/spaCy passate come segnale ausiliario
                      (default: 4 → noun_ratio, verb_ratio, adj_ratio, entity_density)
    dropout         : dropout applicato alle teste

    Forward
    -------
    Input : input_ids (B,T), attention_mask (B,T)
    Output: score (B,1), factuality (B,1), lexical_pred (B, n_lexical_feats), pooled_emb (B,H)
    """

    def __init__(
            self,
            model_name: str = "microsoft/deberta-v3-large",
            n_lexical_feats: int = 4,
            dropout: float = 0.1,
            MODEL_PATH: str = None,
            freeze_encoder: bool = False,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        H = self.encoder.config.hidden_size

        # Forza encoder a FP32
        self.encoder = self.encoder.float()

        if freeze_encoder:
            print(f"[INFO] Congelamento pesi dell'encoder: {model_name}")
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Pooling pesato — cattura meglio le recensioni lunghe
        self.pooling = AttentionPooling(H)

        # ── Score head (testa principale) ──────────────────────────────────
        # Regressione 0-100 per distillare il ragionamento dell'LLM teacher.
        self.score_head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(H // 2, H // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(H // 4, 1),
        )

        # ── Factuality head (testa ausiliaria 1) ───────────────────────────
        # Predice la densità informativa/fattualità (segnale offline, 0-1).
        self.factuality_head = nn.Sequential(
            nn.Linear(H, H // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(H // 4, 1),
        )

        # ── Lexical head (testa ausiliaria 2) ──────────────────────────────
        # Predice n feature lessicali calcolate offline con spaCy.
        # Forza l'encoder ad essere sensibile alle proprietà linguistiche
        # senza richiederle a runtime durante l'inferenza.
        self.lexical_head = nn.Sequential(
            nn.Linear(H, H // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(H // 4, n_lexical_feats),
        )

        self._init_heads()

        if MODEL_PATH is not None and os.path.exists(MODEL_PATH):
            print(f"[INFO] Caricamento modello fine-tuned da {MODEL_PATH}...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
            self.load_state_dict(state_dict)
            print(f"[INFO] Modello caricato da {MODEL_PATH}.")
        elif MODEL_PATH is not None:
            print(f"[WARNING] {MODEL_PATH} non trovato. Uso pesi di default (non addestrato).")
        else:
            # Inizializzazione di default (Xavier) completata silenziosamente
            pass


    # ------------------------------------------------------------------
    def _init_heads(self):
        """Inizializzazione Xavier per tutti i layer lineari delle teste."""
        for module in [self.score_head, self.factuality_head, self.lexical_head,
                       self.pooling]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            **kwargs,
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Attention pooling sull'intera sequenza
        pooled = self.pooling(outputs.last_hidden_state, attention_mask)

        # Cast per gestire dtypes misti (encoder Half, teste Float)
        target_dtype = self.score_head[0].weight.dtype
        pooled = pooled.to(target_dtype)

        # Output normalizzato in [0, 1] per stabilità loss
        score        = torch.sigmoid(self.score_head(pooled))
        factuality   = torch.sigmoid(self.factuality_head(pooled))
        lexical_pred = torch.sigmoid(self.lexical_head(pooled))

        return score, factuality, lexical_pred, pooled


    # ============================================================
    #  Inferenza
    # ============================================================
    def inference_pipeline(
            self,
            text: Union[str, list],
            tokenizer,
            device: Optional[str] = None,
    ) -> Union[float, list]:
        """
        Valuta una o più recensioni restituendo lo score di insightfulness [0-100].

        Parametri
        ----------
        text     : stringa singola o lista di stringhe
        model    : InsightReviewScorer (fine-tuned o default)
        tokenizer: tokenizer HuggingFace compatibile col modello
        device   : 'cuda', 'cpu', o None per auto-detect

        Ritorna
        -------
        float (singola stringa) oppure list[float] (lista di stringhe)
        """
        device    = "cuda" if torch.cuda.is_available() else "cpu"

        single = isinstance(text, str)
        texts  = [text] if single else text

        # Application of prompt template to condition the encoder
        prompt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".prompts", "DeBERTa.txt"))
        with open(prompt_path, "r") as f:
            prompt_tmpl = f.read()
        texts = [prompt_tmpl.format(t) for t in texts]

        self.eval()
        self.to(device)

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            scores, factuality, _, _ = self(**inputs)

        # The model predicts in [0, 1], we map back to [0, 100] for the user
        results = (scores.squeeze(-1) * 100.0).cpu().tolist()

        if single:
            return float(results) if isinstance(results, float) else float(results[0])
        return [float(r) for r in results]


# ============================================================
#  Loss composita
# ============================================================

def compute_insight_loss(
        pred_score:        torch.Tensor,           # (B,) o (B,1) — score predetto
        target_score:      torch.Tensor,           # (B,)          — score LLM teacher
        pred_factuality:   torch.Tensor,           # (B,) o (B,1) — fattualità predetta
        target_factuality: torch.Tensor,           # (B,)          — fattualità offline
        pred_lexical:      torch.Tensor,           # (B, F)        — feature lessicali predette
        target_lexical:    torch.Tensor,           # (B, F)        — feature lessicali offline
        emb:               torch.Tensor,           # (B, H)        — embedding pooled
        emb_worst:         torch.Tensor,           # (B, H)        — bound inferiore categoria
        emb_best:          torch.Tensor,           # (B, H)        — bound superiore categoria
        alpha:  float = 0.50,   # peso L_distill
        beta:   float = 0.15,   # peso L_factuality
        gamma:  float = 0.20,   # peso L_bound
        delta:  float = 0.15,   # peso L_lexical
        margin_w: float = 0.0,  # peso L_rank (0 = disabilitato se batch non ha coppie)
        margin:   float = 5.0,  # margine minimo tra score di coppie diverse
) -> tuple[torch.Tensor, dict]:
    """
    Loss composita a quattro componenti + ranking loss opzionale.

    Componenti
    ----------
    L_distill   (alpha)  : MSE tra score predetto e score LLM teacher (range 0-100).
    L_factuality (beta)  : MSE tra fattualità predetta e segnale offline (range 0-1).
    L_lexical   (delta)  : MSE tra feature lessicali predette e quelle spaCy offline.
                           Forza l'encoder ad essere sensibile alle proprietà linguistiche
                           senza richiedere spaCy a runtime.
    L_bound     (gamma)  : Geometric Bound Loss — la proiezione dell'embedding sull'asse
                           (worst → best) di categoria deve rispecchiare lo score.
    L_rank      (margin_w): Margin Ranking Loss — per ogni coppia (i,j) nello stesso batch
                           con target_score_i > target_score_j + margin, il modello deve
                           predire score_i > score_j.

    Ritorna
    -------
    (total_loss, dict con le singole componenti per il logging)
    """
    pred_score      = pred_score.squeeze()
    pred_factuality = pred_factuality.squeeze()

    # ── 1. Distillation Loss ──────────────────────────────────────────────
    l_distill = F.mse_loss(pred_score, target_score)

    # ── 2. Factuality Loss ───────────────────────────────────────────────
    l_factuality = F.mse_loss(pred_factuality, target_factuality)

    # ── 3. Lexical Feature Loss ──────────────────────────────────────────
    l_lexical = F.mse_loss(pred_lexical, target_lexical)

    # ── 4. Geometric Bound Loss ──────────────────────────────────────────
    v_diff    = emb_best - emb_worst                                    # (B, H)
    v_norm_sq = torch.sum(v_diff ** 2, dim=-1, keepdim=True) + 1e-8    # (B, 1)

    # Maschera sample con bound degeneri (worst ≈ best → categoria con 1 solo campione)
    bound_valid   = (v_norm_sq.squeeze(-1) > 1e-4).float()             # (B,)
    emb_centered  = emb - emb_worst
    scalar_proj   = torch.sum(emb_centered * v_diff, dim=-1) / v_norm_sq.squeeze(-1)   # (B,)
    expected_proj = target_score  # target_score è già normalizzato in [0, 1]
    l_bound_raw   = F.mse_loss(scalar_proj, expected_proj, reduction="none")           # (B,)
    l_bound       = (l_bound_raw * bound_valid).mean()

    # ── 5. Margin Ranking Loss (opzionale) ───────────────────────────────
    if margin_w > 0.0:
        B = pred_score.shape[0]
        # Costruiamo tutte le coppie (i, j) sul device corretto
        i_idx, j_idx = torch.triu_indices(B, B, offset=1, device=pred_score.device)
        diff_target  = target_score[i_idx] - target_score[j_idx]       # (P,)

        # Una coppia è significativa se la differenza ASSOLUTA supera il margine
        valid_pairs  = torch.abs(diff_target) > margin

        if valid_pairs.any():
            valid_i = i_idx[valid_pairs]
            valid_j = j_idx[valid_pairs]

            si = pred_score[valid_i]
            sj = pred_score[valid_j]

            # Target per ranking: 1 se target_i > target_j, altrimenti -1
            target_y = torch.sign(diff_target[valid_pairs])

            l_rank  = F.margin_ranking_loss(si, sj, target_y, margin=margin)
        else:
            l_rank = torch.tensor(0.0, device=pred_score.device)
    else:
        l_rank = torch.tensor(0.0, device=pred_score.device)

    # ── Totale pesato ────────────────────────────────────────────────────
    total = (
            alpha    * l_distill    +
            beta     * l_factuality +
            gamma    * l_bound      +
            delta    * l_lexical    +
            margin_w * l_rank
    )

    components = {
        "l_distill":    l_distill.item(),
        "l_factuality": l_factuality.item(),
        "l_lexical":    l_lexical.item(),
        "l_bound":      l_bound.item(),
        "l_rank":       l_rank.item(),
        "total":        total.item(),
    }

    return total, components




# ============================================================
#  Entry point / demo
# ============================================================

if __name__ == "__main__":
    MODEL_PATH = "../.weights/v7_frozen/best.pt"
    MODEL_NAME = "microsoft/deberta-v3-large"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = InsightReviewScorer(MODEL_NAME,MODEL_PATH=MODEL_PATH)


    review_alta = (
        "These Cressi Saint-Tropez flip-flops were a wonderful discovery, "
        "revealing themselves to be a well-made product that combines the brand's reliability "
        "with a modern and vibrant design. The first thing that strikes you is the color combination, "
        "especially the light blue shade: it's a fresh, bright, summery color that really stands out. Although they're designed for girls and teenagers, the fit is "
        "generous and versatile, making it perfect even for an adult foot ranging from "
        "36 to 37, ensuring natural and stable support. "
        "From a technical standpoint, the quality of the materials is evident. The rubber footbed is "
        "compact and supports the foot well without giving way at the heel, while the Y-strap is "
        "firmly embedded in the sole, giving a feeling of superior sturdiness "
        "than cheaper models. They're incredibly light and take up very little space "
        "in your bag, but they don't sacrifice safety: the sole The rough surface offers excellent non-slip grip, essential for moving fearlessly on wet surfaces. The comfort is absolute even after many hours of use; the toe separator is soft and thin, allowing you to use them for long walks on the beach without the risk of annoying blisters. Made from water-resistant materials, they dry in a flash and are very easy to clean. Although the price may seem slightly higher than average, the durability and attention to detail fully justify the investment."
    )
    review_alta=(
        "The headphones sound clear at medium volume, but the bass distorts above 80%. "
        "Battery lasted about 18 hours over three days of commuting, and the ear pads "
        "became warm after one hour."
    )
    review_bassa = "A"


    score_high = model.inference_pipeline(review_alta, tokenizer)
    score_low  = model.inference_pipeline(review_bassa, tokenizer)

    print(f"\nModel: {'Fine-tuned' if os.path.exists(MODEL_PATH) else 'Default (non addestrato)'}")
    print(f"Insight Score (Alta qualità) : {score_high:.2f}/100")
    print(f"Insight Score (Bassa qualità): {score_low:.2f}/100")


