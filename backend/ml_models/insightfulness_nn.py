"""
Architecture
------------
  Encoder (DeBERTa-v3-large)
      |
      ├─ AttentionPooling        → weighted global representation (better than [CLS] alone
      │                            for long and structured reviews)
      │
      ├─ score_head              → 0-100 regression (LLM teacher distillation)   [alpha]
      ├─ factuality_head         → offline information density/factuality           [beta]
      └─ lexical_head            → spaCy features fused into the latent space         [delta]

Composite Loss
--------------
  L = alpha * L_distill   (MSE score vs LLM teacher)
    + beta  * L_factuality (MSE predicted factuality vs offline signal)
    + gamma * L_bound      (geometric alignment worst→best per category)
    + delta * L_rank       (margin ranking: review_A > review_B if score_A > score_B)

Inference
---------
  No dependency on spaCy or category data at runtime:
  the encoder learns everything it needs from the raw text.
  spaCy features are used only as supervision signals during training.
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
    Weighted pooling over the entire token sequence via an attention scorer.
    Returns a (B, H) representation that better captures long reviews
    compared to the [CLS] token alone.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(
            self,
            hidden_states: torch.Tensor,        # (B, T, H)
            attention_mask: torch.Tensor,       # (B, T)
    ) -> torch.Tensor:                      # (B, H)
        # Attention score for each token
        scores = self.attn(hidden_states).squeeze(-1)           # (B, T)

        # Mask padding with -inf before softmax
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)   # (B, T, 1)

        pooled = (hidden_states * weights).sum(dim=1)           # (B, H)
        return pooled


# ============================================================
#  Main Model
# ============================================================

class InsightReviewScorer(nn.Module):
    """
    End-to-end scorer for review insightfulness.

    Parameters
    ----------
    model_name      : HuggingFace encoder model name (default: deberta-v3-large)
    n_lexical_feats : number of lexical/spaCy features passed as auxiliary signal
                      (default: 4 → noun_ratio, verb_ratio, adj_ratio, entity_density)
    dropout         : dropout applied to the heads

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

        # Force encoder to FP32
        self.encoder = self.encoder.float()

        if freeze_encoder:
            print(f"[INFO] Freezing encoder weights: {model_name}")
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Weighted pooling — captures long reviews better
        self.pooling = AttentionPooling(H)

        # ── Score head (main head) ──────────────────────────────────
        # 0-100 regression to distill LLM teacher reasoning.
        self.score_head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(H // 2, H // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(H // 4, 1),
        )

        # ── Factuality head (auxiliary head 1) ────
        # Predicts information density/factuality (offline signal, 0-1).
        self.factuality_head = nn.Sequential(
            nn.Linear(H, H // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(H // 4, 1),
        )

        # ── Lexical head (auxiliary head 2) ─────
        # Predicts n lexical features calculated offline with spaCy.
        # Forces the encoder to be sensitive to linguistic properties
        # without requiring them at runtime during inference.
        self.lexical_head = nn.Sequential(
            nn.Linear(H, H // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(H // 4, n_lexical_feats),
        )

        self._init_heads()

        if MODEL_PATH is not None and os.path.exists(MODEL_PATH):
            print(f"[INFO] Loading fine-tuned model from {MODEL_PATH}...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
            self.load_state_dict(state_dict)
            print(f"[INFO] Model loaded from {MODEL_PATH}.")
        elif MODEL_PATH is not None:
            print(f"[WARNING] {MODEL_PATH} not found. Using default weights (untrained).")
        else:
            # Default initialization (Xavier) completed silently
            pass


    # ------------------------------------------------------------------
    def _init_heads(self):
        """Xavier initialization for all linear layers of the heads."""
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

        # Attention pooling over the entire sequence
        pooled = self.pooling(outputs.last_hidden_state, attention_mask)

        # Cast to handle mixed dtypes (Half encoder, Float heads)
        target_dtype = self.score_head[0].weight.dtype
        pooled = pooled.to(target_dtype)

        # Output normalized in [0, 1] for loss stability
        score        = torch.sigmoid(self.score_head(pooled))
        factuality   = torch.sigmoid(self.factuality_head(pooled))
        lexical_pred = torch.sigmoid(self.lexical_head(pooled))

        return score, factuality, lexical_pred, pooled


    # ============================================================
    #  Inference
    # ============================================================
    def inference_pipeline(
            self,
            text: Union[str, list],
            tokenizer,
            device: Optional[str] = None,
    ) -> Union[float, list]:
        """
        Evaluates one or more reviews returning the insightfulness score [0-100].

        Parameters
        ----------
        text     : single string or list of strings
        model    : InsightReviewScorer (fine-tuned or default)
        tokenizer: HuggingFace tokenizer compatible with the model
        device   : 'cuda', 'cpu', or None for auto-detect

        Returns
        -------
        float (single string) or list[float] (list of strings)
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
#  Composite Loss
# ============================================================

def compute_insight_loss(
        pred_score:        torch.Tensor,           # (B,) or (B,1) — predicted score
        target_score:      torch.Tensor,           # (B,)          — LLM teacher score
        pred_factuality:   torch.Tensor,           # (B,) or (B,1) — predicted factuality
        target_factuality: torch.Tensor,           # (B,)          — offline factuality
        pred_lexical:      torch.Tensor,           # (B, F)        — predicted lexical features
        target_lexical:    torch.Tensor,           # (B, F)        — offline lexical features
        emb:               torch.Tensor,           # (B, H)        — pooled embedding
        emb_worst:         torch.Tensor,           # (B, H)        — category lower bound
        emb_best:          torch.Tensor,           # (B, H)        — category upper bound
        alpha:  float = 0.50,   # L_distill weight
        beta:   float = 0.15,   # L_factuality weight
        gamma:  float = 0.20,   # L_bound weight
        delta:  float = 0.15,   # L_lexical weight
        margin_w: float = 0.0,  # L_rank weight (0 = disabled if batch has no pairs)
        margin:   float = 5.0,  # minimum margin between scores of different pairs
) -> tuple[torch.Tensor, dict]:
    """
    Four-component composite loss + optional ranking loss.

    Components
    ----------
    L_distill   (alpha)  : MSE between predicted score and LLM teacher score (range 0-100).
    L_factuality (beta)  : MSE between predicted factuality and offline signal (range 0-1).
    L_lexical   (delta)  : MSE between predicted lexical features and offline spaCy ones.
                           Forces the encoder to be sensitive to linguistic properties
                           without requiring spaCy at runtime.
    L_bound     (gamma)  : Geometric Bound Loss — the projection of the embedding onto the
                           (worst → best) category axis must reflect the score.
    L_rank      (margin_w): Margin Ranking Loss — for each pair (i,j) in the same batch
                           with target_score_i > target_score_j + margin, the model must
                           predict score_i > score_j.

    Returns
    -------
    (total_loss, dict with individual components for logging)
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

    # Mask samples with degenerate bounds (worst ≈ best → category with only 1 sample)
    bound_valid   = (v_norm_sq.squeeze(-1) > 1e-4).float()             # (B,)
    emb_centered  = emb - emb_worst
    scalar_proj   = torch.sum(emb_centered * v_diff, dim=-1) / v_norm_sq.squeeze(-1)   # (B,)
    expected_proj = target_score  # target_score is already normalized in [0, 1]
    l_bound_raw   = F.mse_loss(scalar_proj, expected_proj, reduction="none")           # (B,)
    l_bound       = (l_bound_raw * bound_valid).mean()

    # ── 5. Margin Ranking Loss (optional) ───────────────────────────────
    if margin_w > 0.0:
        B = pred_score.shape[0]
        # Build all pairs (i, j) on the correct device
        i_idx, j_idx = torch.triu_indices(B, B, offset=1, device=pred_score.device)
        diff_target  = target_score[i_idx] - target_score[j_idx]       # (P,)

        # A pair is significant if the ABSOLUTE difference exceeds the margin
        valid_pairs  = torch.abs(diff_target) > margin

        if valid_pairs.any():
            valid_i = i_idx[valid_pairs]
            valid_j = j_idx[valid_pairs]

            si = pred_score[valid_i]
            sj = pred_score[valid_j]

            # Target for ranking: 1 if target_i > target_j, otherwise -1
            target_y = torch.sign(diff_target[valid_pairs])

            l_rank  = F.margin_ranking_loss(si, sj, target_y, margin=margin)
        else:
            l_rank = torch.tensor(0.0, device=pred_score.device)
    else:
        l_rank = torch.tensor(0.0, device=pred_score.device)

    # ── Weighted Total ────────────────────────────────────────────────────
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

    print(f"\nModel: {'Fine-tuned' if os.path.exists(MODEL_PATH) else 'Default (untrained)'}")
    print(f"Insight Score (High quality) : {score_high:.2f}/100")
    print(f"Insight Score (Low quality): {score_low:.2f}/100")
