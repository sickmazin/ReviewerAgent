"""
train_insight_model.py
======================
Pipeline di training per InsightReviewScorer (insight_model.py).

Feature principali
------------------
  - Feature lessicali spaCy calcolate offline e usate come supervisione (lexical_head)
  - Category bounds cachati su disco (non ricalcolati ad ogni run)
  - LR Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
  - Validation split (20%) con early stopping (patience configurabile)
  - Gradient accumulation
  - Margin Ranking Loss abilitata se il batch contiene coppie significative
  - Logging: train_loss, val_loss, MAE, Pearson R, ogni singola componente di loss
  - Checkpoint salvato SOLO se val_loss migliora (no spam su disco)
  - Resume automatico dall'ultimo checkpoint numerato

CSV atteso
----------
  Colonne obbligatorie : text, insight_score, category
  insight_score        : float in [0, 100] (etichette LLM teacher)
  category             : stringa (es. "scarpe", "elettronica", ...)
"""

import glob
import os
import re
import time
from typing import Optional

import numpy as np
import pandas as pd
import spacy
import torch
import torch.optim as optim
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from transformers import AutoTokenizer

from insight_nn_v2 import InsightReviewScorer, compute_insight_loss


# ============================================================
#  spaCy — caricamento con fallback
# ============================================================

def _load_spacy() -> spacy.language.Language:
    """
    Carica it_core_news_sm (italiano) con fallback su en_core_web_sm.
    Parser e NER disabilitati per velocità (usiamo solo il tagger POS).
    """
    for name in ["en_core_web_sm", "it_core_news_sm"]:
        try:
            return spacy.load(name, disable=["ner", "parser"])
        except OSError:
            os.system(f"python -m spacy download {name}")
            try:
                return spacy.load(name, disable=["ner", "parser"])
            except OSError:
                continue
    raise RuntimeError("Nessun modello spaCy trovato. Installa: python -m spacy download it_core_news_sm")


nlp = _load_spacy()


# ============================================================
#  Feature lessicali offline
# ============================================================

def extract_lexical_features(text: str) -> list:
    """
    Estrae 4 feature lessicali normalizzate in [0, 1]:
      [0] noun_ratio    — (NOUN + PROPN) / tot
      [1] verb_ratio    — VERB / tot
      [2] adj_ratio     — ADJ / tot
      [3] entity_density— entità named / tot

    Queste feature vengono usate come target della lexical_head durante il training,
    forzando l'encoder ad essere sensibile alle proprietà linguistiche del testo
    senza richiedere spaCy a runtime durante l'inferenza.
    """
    doc   = nlp(str(text))
    tot   = max(len(doc), 1)
    nouns = sum(1 for t in doc if t.pos_ in {"NOUN", "PROPN"})
    verbs = sum(1 for t in doc if t.pos_ == "VERB")
    adjs  = sum(1 for t in doc if t.pos_ == "ADJ")
    ents  = len(doc.ents)

    return [
        float(np.clip(nouns / tot, 0, 1)),
        float(np.clip(verbs / tot, 0, 1)),
        float(np.clip(adjs  / tot, 0, 1)),
        float(np.clip(ents  / tot, 0, 1)),
    ]


def extract_factuality_signal(text: str) -> float:
    """
    Rapporto (NOUN + PROPN + VERB) / (ADJ + 2) come proxy di fattualità.
    Valore in [0, 1].
    """
    doc   = nlp(str(text))
    nouns = sum(1 for t in doc if t.pos_ in {"NOUN", "PROPN"})
    verbs = sum(1 for t in doc if t.pos_ == "VERB")
    adjs  = sum(1 for t in doc if t.pos_ == "ADJ")
    return float(min((nouns + verbs) / (adjs + 2.0), 1.0))


def preprocess_texts(texts: list, pbar: Optional[tqdm] = None) -> tuple[list, list]:
    """
    Calcola in parallelo factuality_signal e lexical_features per una lista di testi.
    Ritorna (factuality_signals: list[float], lexical_features: list[list[float]]).
    """
    num_proc = 1 if os.name == "nt" else min(os.cpu_count() or 1, 4)
    print(f"Preprocessing spaCy su {num_proc} core...")

    factuality_signals = []
    lexical_features   = []

    try:
        for doc in nlp.pipe(texts, n_process=num_proc, batch_size=64):
            tot   = max(len(doc), 1)
            nouns = sum(1 for t in doc if t.pos_ in {"NOUN", "PROPN"})
            verbs = sum(1 for t in doc if t.pos_ == "VERB")
            adjs  = sum(1 for t in doc if t.pos_ == "ADJ")
            ents  = len(doc.ents)

            factuality_signals.append(float(min((nouns + verbs) / (adjs + 2.0), 1.0)))
            lexical_features.append([
                float(np.clip(nouns / tot, 0, 1)),
                float(np.clip(verbs / tot, 0, 1)),
                float(np.clip(adjs  / tot, 0, 1)),
                float(np.clip(ents  / tot, 0, 1)),
            ])
            if pbar:
                pbar.update(1)

    except Exception as exc:
        print(f"[WARNING] Fallback sequenziale: {exc}")
        for t in tqdm(texts, desc="Preprocessing"):
            factuality_signals.append(extract_factuality_signal(t))
            lexical_features.append(extract_lexical_features(t))

    return factuality_signals, lexical_features


# ============================================================
#  Dataset
# ============================================================

N_LEXICAL_FEATS = 4     # deve corrispondere a n_lexical_feats in InsightReviewScorer


class ReviewDataset(Dataset):
    """
    Dataset PyTorch per InsightReviewScorer.

    Colonne richieste nel DataFrame:
      text, insight_score, category, factuality_signal,
      lex_noun, lex_verb, lex_adj, lex_ent
    """

    def __init__(
            self,
            df: pd.DataFrame,
            tokenizer,
            category_bounds: dict,
            max_length: int = 512,
    ):
        self.texts      = df["text"].astype(str).tolist()
        self.scores     = df["insight_score"].values.astype(np.float32)
        self.factuality = df["factuality_signal"].values.astype(np.float32)
        self.lexical    = df[["lex_noun", "lex_verb", "lex_adj", "lex_ent"]].values.astype(np.float32)
        self.categories = df["category"].values.tolist()
        self.tokenizer  = tokenizer
        self.bounds     = category_bounds
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        cat    = self.categories[idx]
        bounds = self.bounds[cat]

        return {
            "input_ids":      enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "target_score":   torch.tensor(self.scores[idx],    dtype=torch.float32),
            "target_fact":    torch.tensor(self.factuality[idx], dtype=torch.float32),
            "target_lexical": torch.tensor(self.lexical[idx],   dtype=torch.float32),
            "emb_worst":      bounds["worst"],
            "emb_best":       bounds["best"],
        }


# ============================================================
#  Utilità
# ============================================================

def _get_embedding(text: str, model: InsightReviewScorer, tokenizer, device: str) -> torch.Tensor:
    """Embedding pooled per un singolo testo, su CPU in float32."""
    inputs = tokenizer(
        str(text), return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        _, _, _, emb = model(**inputs)
    return emb.cpu().squeeze(0).to(torch.float32)


def _compute_category_bounds(
        df: pd.DataFrame,
        model: InsightReviewScorer,
        tokenizer,
        device: str,
        cache_path: str,
) -> dict:
    """
    Calcola (o carica dalla cache) gli embedding di bound per ogni categoria.
    I bound vengono salvati su disco: non vengono ricalcolati ad ogni run.
    Attenzione: se il modello cambia significativamente, elimina il file cache.
    """
    if os.path.exists(cache_path):
        print(f"Category bounds dalla cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    print("Calcolo category bounds...")
    model.eval()
    bounds = {}
    for cat in tqdm(df["category"].unique(), desc="Categorie"):
        cat_df    = df[df["category"] == cat]
        idx_worst = cat_df["insight_score"].idxmin()
        idx_best  = cat_df["insight_score"].idxmax()
        bounds[cat] = {
            "worst": _get_embedding(df.loc[idx_worst, "text"], model, tokenizer, device),
            "best":  _get_embedding(df.loc[idx_best,  "text"], model, tokenizer, device),
        }

    torch.save(bounds, cache_path)
    print(f"Category bounds salvati in {cache_path}")
    return bounds


def _evaluate(
        model: InsightReviewScorer,
        loader: DataLoader,
        device: str,
        loss_weights: dict,
) -> tuple:
    """
    Valuta il modello sulla validation set.

    Ritorna
    -------
    (avg_loss, mae, pearson_r, avg_components: dict)
    """
    model.eval()
    total_loss  = 0.0
    all_preds, all_targets = [], []
    comp_accum  = {}

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            target_score   = batch["target_score"].to(device, non_blocking=True)
            target_fact    = batch["target_fact"].to(device, non_blocking=True)
            target_lexical = batch["target_lexical"].to(device, non_blocking=True)
            emb_worst      = batch["emb_worst"].to(device, non_blocking=True)
            emb_best       = batch["emb_best"].to(device, non_blocking=True)

            pred_score, pred_fact, pred_lex, cls_emb = model(input_ids, attention_mask)
            loss, comps = compute_insight_loss(
                pred_score, target_score,
                pred_fact,  target_fact,
                pred_lex,   target_lexical,
                cls_emb,    emb_worst, emb_best,
                **loss_weights,
            )

            total_loss += loss.item()
            all_preds.extend(pred_score.squeeze().cpu().tolist())
            all_targets.extend(target_score.cpu().tolist())
            for k, v in comps.items():
                comp_accum[k] = comp_accum.get(k, 0.0) + v

    n          = max(len(loader), 1)
    avg_loss   = total_loss / n
    mae        = float(np.mean(np.abs(np.array(all_preds) - np.array(all_targets))))
    r, _       = pearsonr(all_preds, all_targets) if len(all_preds) > 2 else (0.0, None)
    avg_comps  = {k: v / n for k, v in comp_accum.items()}

    return avg_loss, mae, float(r), avg_comps


# ============================================================
#  Training principale
# ============================================================

def train_model(
        csv_path:               str = "dataset/reviews_labeled.csv",
        model_name:             str   = "microsoft/deberta-v3-large",
        epochs:                 int   = 100,
        batch_size:             int   = 16,
        lr:                     float = 1e-5,
        accumulation_steps:     int   = 2,
        val_split:              float = 0.2,
        early_stopping_patience: int  = 10,
        checkpoint_dir:         str   = "models/v3",
        load_model:             str   = None,
        # Pesi loss (devono sommare a 1 + margin_w)
        alpha:    float = 0.50,
        beta:     float = 0.15,
        gamma:    float = 0.20,
        delta:    float = 0.15,
        margin_w: float = 0.10,
        margin:   float = 5.0,
):
    """
    Addestra InsightReviewScorer sul CSV specificato.

    Parametri principali
    --------------------
    csv_path      : path al CSV (colonne: text, insight_score, category)
    model_name    : encoder HuggingFace
    epochs        : epoche massime
    batch_size    : dimensione batch (effettivo = batch_size * accumulation_steps)
    lr            : learning rate iniziale
    val_split     : frazione validation
    early_stopping_patience : epoche senza miglioramento prima dello stop
    alpha/beta/gamma/delta/margin_w : pesi della loss composita
    """

    loss_weights = dict(
        alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        margin_w=margin_w, margin=margin,
    )

    # ------------------------------------------------------------------
    # 1. Caricamento / preprocessing CSV
    # ------------------------------------------------------------------
    processed_csv = csv_path.replace(".csv", "_processed.csv")

    if os.path.exists(processed_csv):
        print(f"Dataset preprocessato trovato: {processed_csv}")
        df = pd.read_csv(processed_csv)
    else:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["text", "insight_score"])
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"] != ""].reset_index(drop=True)

        print("Calcolo feature spaCy offline...")
        with tqdm(total=len(df), desc="spaCy") as pbar:
            facts, lexicals = preprocess_texts(df["text"].tolist(), pbar)

        df["factuality_signal"] = facts
        lex_arr = np.array(lexicals, dtype=np.float32)
        df["lex_noun"] = lex_arr[:, 0]
        df["lex_verb"] = lex_arr[:, 1]
        df["lex_adj"]  = lex_arr[:, 2]
        df["lex_ent"]  = lex_arr[:, 3]

        df.to_csv(processed_csv, index=False)
        print(f"Dataset salvato in {processed_csv}")

    required_cols = ["text", "insight_score", "category",
        "factuality_signal", "lex_noun", "lex_verb", "lex_adj", "lex_ent"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    print(f"Dataset: {len(df)} righe, {df['category'].nunique()} categorie")

    # ------------------------------------------------------------------
    # 2. Device e modello
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = InsightReviewScorer(model_name, n_lexical_feats=N_LEXICAL_FEATS).to(device)
    model     = model.float()  # Forza tutto a FP32
    if load_model is not None:
        try:
            model.load_state_dict(torch.load(load_model, map_location=device, weights_only=True))
            print("[MODEL LOADED] ", load_model)
        except RuntimeError as e:
            print(f"[WARNING] Errore caricamento pesi: {e}. Continuo con pesi casuali.")
            # Se mismatch, non caricare e continuare con pesi init
    # ------------------------------------------------------------------
    # 3. Cartelle e path
    # ------------------------------------------------------------------
    os.makedirs(checkpoint_dir, exist_ok=True)
    history_path   = os.path.join(checkpoint_dir, "training_history.csv")
    bounds_cache   = os.path.join(checkpoint_dir, "bounds_cache.pt")
    best_ckpt_path = os.path.join(checkpoint_dir, "best.pt")

    # ------------------------------------------------------------------
    # 4. Resume da checkpoint (solo se non si carica un modello specifico)
    # ------------------------------------------------------------------
    start_epoch      = 0
    best_val_loss    = float("inf")
    patience_counter = 0
    history          = []

    if load_model is None:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt"))
        if checkpoints:
            epochs_found = [int(re.search(r"epoch_(\d+)\.pt", f).group(1)) for f in checkpoints]
            latest_epoch = max(epochs_found)
            ckpt_path    = os.path.join(checkpoint_dir, f"epoch_{latest_epoch}.pt")
            print(f"Resume da checkpoint: {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            start_epoch = latest_epoch

            if os.path.exists(history_path):
                history      = pd.read_csv(history_path).to_dict("records")
                past_vals    = [h.get("val_loss", float("inf")) for h in history]
                best_val_loss = min(past_vals) if past_vals else float("inf")
                print(f"History: {len(history)} epoche | best val_loss: {best_val_loss:.4f}")
    else:
        print(f"Caricato modello specifico: {load_model}, resume disabilitato.")

    # ------------------------------------------------------------------
    # 5. Category bounds (cachati su disco)
    # ------------------------------------------------------------------
    category_bounds = _compute_category_bounds(df, model, tokenizer, device, bounds_cache)

    # ------------------------------------------------------------------
    # 6. Dataset + DataLoader
    # ------------------------------------------------------------------
    full_ds   = ReviewDataset(df, tokenizer, category_bounds)
    val_size  = int(len(full_ds) * val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    nw = 0  # Impostato a 0 per evitare errori di Access Violation (0xC0000005) su Windows
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=(device == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=nw, pin_memory=(device == "cuda"),
    )
    print(f"Train: {train_size} | Val: {val_size} | Workers: {nw}")

    # ------------------------------------------------------------------
    # 7. Ottimizzatore, scheduler, scaler
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, eps=1e-8)

    # Cosine LR con warm restarts: T_0=10 epoche, poi cicli doppi
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=2, T_mult=2, eta_min=1e-9,
    )

    # ------------------------------------------------------------------
    # 8. Loop di training
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        t0          = time.perf_counter()
        model.train()
        epoch_loss  = 0.0
        valid_steps = 0
        optimizer.zero_grad()

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for i, batch in enumerate(loop):
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            target_score   = batch["target_score"].to(device, non_blocking=True)
            target_fact    = batch["target_fact"].to(device, non_blocking=True)
            target_lexical = batch["target_lexical"].to(device, non_blocking=True)
            emb_worst      = batch["emb_worst"].to(device, non_blocking=True)
            emb_best       = batch["emb_best"].to(device, non_blocking=True)

            pred_score, pred_fact, pred_lex, cls_emb = model(input_ids, attention_mask)
            loss, _ = compute_insight_loss(
                pred_score, target_score,
                pred_fact,  target_fact,
                pred_lex,   target_lexical,
                cls_emb,    emb_worst, emb_best,
                **loss_weights,
            )

            loss = loss / accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARNING] Loss NaN/Inf allo step {i}, batch saltato.")
                continue

            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()


            step_loss   = loss.item() * accumulation_steps
            epoch_loss  += step_loss
            valid_steps += 1
            loop.set_postfix(loss=f"{step_loss:.4f}")

        scheduler.step()

        # ------------------------------------------------------------------
        # 9. Validation
        # ------------------------------------------------------------------
        val_loss, val_mae, val_r, val_comps = _evaluate(
            model, val_loader, device, loss_weights
        )

        epoch_dur      = time.perf_counter() - t0
        avg_train_loss = epoch_loss / max(valid_steps, 1)
        current_lr     = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:3d} | "
            f"train: {avg_train_loss:.4f} | "
            f"val: {val_loss:.4f} | "
            f"MAE: {val_mae:.2f} | "
            f"R: {val_r:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{epoch_dur:.1f}s"
        )
        # Log componenti loss validation
        comp_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_comps.items())
        print(f"  └─ {comp_str}")

        # ------------------------------------------------------------------
        # 10. Logging su CSV
        # ------------------------------------------------------------------
        row = {
            "epoch": epoch + 1, "train_loss": avg_train_loss,
            "val_loss": val_loss, "val_mae": val_mae, "val_r": val_r,
            "lr": current_lr, "duration": epoch_dur,
        }
        row.update({f"val_{k}": v for k, v in val_comps.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        # ------------------------------------------------------------------
        # 11. Checkpoint + Early Stopping
        # ------------------------------------------------------------------
        torch.save(
            model.state_dict(),
            os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt"),
        )
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt"),
            )
            print(f"  ✓ Nuovo best val_loss: {best_val_loss:.4f} → {best_ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping dopo {epoch+1} epoche (patience={early_stopping_patience}).")
                break
            print(f"  · No improvement ({patience_counter}/{early_stopping_patience})")

    # ------------------------------------------------------------------
    # 12. Salvataggio finale
    # ------------------------------------------------------------------
    final_path = os.path.join(checkpoint_dir, "final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining completato.")
    print(f"  Modello finale : {final_path}")
    print(f"  Miglior modello: {best_ckpt_path}  (val_loss: {best_val_loss:.4f})")


# ============================================================
#  Entry point
# ============================================================

if __name__ == "__main__":
    train_model(
        csv_path                ="../../.datasets/reviews_labeled.csv",
        model_name              = "microsoft/deberta-v3-large",
        epochs                  = 100,
        batch_size              = 4,
        lr                      = 2e-7,
        accumulation_steps      = 20,
        val_split               = 0.2,
        early_stopping_patience = 15,
        checkpoint_dir          = "../../.models/v4",
        load_model              = None,
        # Loss weights
        alpha    = 0.50,    # distillation
        beta     = 0.15,    # factuality
        gamma    = 0.20,    # geometric bound
        delta    = 0.15,    # lexical features
        margin_w = 0.10,    # ranking loss (mettere 0.0 per disabilitare)
        margin   = 5.0,     # soglia minima tra coppie per la ranking loss
    )
