"""
training.py
===========
Training pipeline for InsightReviewScorer (insight_model.py).

Key Features
------------
  - spaCy lexical features calculated offline and used as supervision (lexical_head)
  - Category bounds cached on disk (not recalculated on every run)
  - LR Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
  - Validation split (20%) with early stopping (configurable patience)
  - Gradient accumulation
  - Margin Ranking Loss enabled if the batch contains significant pairs
  - Logging: train_loss, val_loss, MAE, Pearson R, each individual loss component
  - Checkpoint saved ONLY if val_loss improves (no disk spam)
  - Automatic resume from the last numbered checkpoint

Expected CSV
------------
  Mandatory columns : text, insight_score, category
  insight_score     : float in [0, 100] (LLM teacher labels)
  category          : string (e.g., "shoes", "electronics", ...)
"""

import glob
import os
import re
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import SubsetRandomSampler

from dataset.dataset import N_LEXICAL_FEATS, ReviewDataset
from insightfulness_nn import InsightReviewScorer, compute_insight_loss

# ============================================================
#  Main Training
# ============================================================

def train_model(
        csv_path:               str = "../.dataset/reviews_labeled.csv",
        model_name:             str   = "microsoft/deberta-v3-large",
        epochs:                 int   = 100,
        batch_size:             int   = 16,
        lr:                     float = 1e-6,
        accumulation_steps:     int   = 2,
        val_split:              float = 0.2,
        early_stopping_patience: int  = 15,
        checkpoint_dir:         str   = "models/v3",
        load_model:             str   = None,
        freeze_encoder:         bool  = False,
        data_samples:            int   = 30000,
        # Loss weights (must sum to 1 + margin_w)
        alpha:    float = 0.50,
        beta:     float = 0.15,
        gamma:    float = 0.20,
        delta:    float = 0.15,
        margin_w: float = 0.10,
        margin:   float = 5.0,
):
    """
    Trains InsightReviewScorer on the specified CSV.

    Main Parameters
    ---------------
    csv_path      : path to CSV (columns: text, insight_score, category)
    model_name    : HuggingFace encoder
    epochs        : maximum epochs
    batch_size    : batch size (effective = batch_size * accumulation_steps)
    lr            : initial learning rate
    val_split     : validation fraction
    early_stopping_patience : epochs without improvement before stopping
    alpha/beta/gamma/delta/margin_w : composite loss weights
    """

    loss_weights = dict(
        alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        margin_w=margin_w, margin=margin,
    )

    # ------------------------------------------------------------------
    # 2. Device and model
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = InsightReviewScorer(
        model_name, 
        n_lexical_feats=N_LEXICAL_FEATS, 
        freeze_encoder=freeze_encoder
    ).to(device)
    model     = model.float()  # Force all to FP32
    if load_model is not None:
        try:
            model.load_state_dict(torch.load(load_model, map_location=device, weights_only=True))
            print(f"[MODEL LOADED] Pre-trained weights successfully loaded from: {load_model} (initial random weights overwritten)")
        except RuntimeError as e:
            print(f"[WARNING] Error loading weights from {load_model}: {e}. Continuing with random weights.")
            # If mismatch, do not load and continue with init weights
            
    # ------------------------------------------------------------------
    # 3. Directories and paths
    # ------------------------------------------------------------------
    os.makedirs(checkpoint_dir, exist_ok=True)
    history_path   = os.path.join(checkpoint_dir, "training_history.csv")
    bounds_cache   = os.path.join(checkpoint_dir, "bounds_cache.pt")
    best_ckpt_path = os.path.join(checkpoint_dir, "best.pt")

    # ------------------------------------------------------------------
    # 4. Resume from checkpoint (only if not loading a specific model and if folder contains weights)
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
            print(f"Resuming from checkpoint: {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            start_epoch = latest_epoch

            if os.path.exists(history_path):
                history      = pd.read_csv(history_path).to_dict("records")
                past_vals    = [h.get("val_loss", float("inf")) for h in history]
                best_val_loss = min(past_vals) if past_vals else float("inf")
                print(f"History: {len(history)} epochs | best val_loss: {best_val_loss:.4f}")
    else:
        print(f"Specific model loaded: {load_model}, resume disabled.")


    # ------------------------------------------------------------------
    # 6. Dataset + DataLoader
    # ------------------------------------------------------------------
    full_ds   = ReviewDataset(
        tokenizer,
        model,
        device,
        bounds_cache,
        csv_path
    )

    val_size  = int(len(full_ds) * val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    nw = 0  # 0 avoids Access Violation on Windows
    samples_per_epoch = min(data_samples, len(train_ds))
    val_samples_per_epoch = int(samples_per_epoch * val_split) # proportional to training (e.g., 20%)
    print(f"Train: {train_size} (sampled {samples_per_epoch}/epoch) | Val: {val_size} (sampled {val_samples_per_epoch}/epoch)")

    # ------------------------------------------------------------------
    # 7. Optimizer, scheduler, scaler
    # ------------------------------------------------------------------
    # Filter parameters: only those with requires_grad=True are optimized
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2, eps=1e-8)

    steps_per_epoch = max(1, samples_per_epoch // batch_size)
    # Cosine LR with warm restarts: decreases every batch and restarts at the next epoch (1-epoch period)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=steps_per_epoch, T_mult=1, eta_min=1e-8,
    )

    # ------------------------------------------------------------------
    # 8. Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        # Dynamic sampling: 50k DIFFERENT random samples for each epoch
        sampler = SubsetRandomSampler(
            torch.randperm(len(train_ds))[:samples_per_epoch].tolist()
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=nw, pin_memory=(device == "cuda"), drop_last=True,
        )

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
                print(f"[WARNING] Loss NaN/Inf at step {i}, skipping batch.")
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
        val_sampler = SubsetRandomSampler(
            torch.randperm(len(val_ds))[:val_samples_per_epoch].tolist()
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size * 2, sampler=val_sampler,
            num_workers=nw, pin_memory=(device == "cuda"), drop_last=False,
        )

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
        # Log validation loss components
        comp_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_comps.items())
        print(f"  └─ {comp_str}")

        # ------------------------------------------------------------------
        # 10. CSV Logging
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
            print(f"  ✓ New best val_loss: {best_val_loss:.4f} → {best_ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping after {epoch+1} epochs (patience={early_stopping_patience}).")
                break
            print(f"  · No improvement ({patience_counter}/{early_stopping_patience})")

    # ------------------------------------------------------------------
    # 12. Final save
    # ------------------------------------------------------------------
    final_path = os.path.join(checkpoint_dir, "final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete.")
    print(f"  Final model : {final_path}")
    print(f"  Best model  : {best_ckpt_path}  (val_loss: {best_val_loss:.4f})")


# ============================================================
#  Validation
# ============================================================

def _evaluate(
        model: InsightReviewScorer,
        loader: DataLoader,
        device: str,
        loss_weights: dict,
) -> tuple:
    """
    Evaluates the model on the validation set.

    Returns
    -------
    (avg_loss, mae, pearson_r, avg_components: dict)
    """
    model.eval()
    total_loss  = 0.0
    all_preds, all_targets = [], []
    comp_accum  = {}

    with torch.no_grad():
        loop = tqdm(loader, desc="Validation", leave=False)
        for batch in loop:
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

            loop.set_postfix(loss=f"{loss.item():.4f}")

    n          = max(len(loader), 1)
    avg_loss   = total_loss / n
    mae        = float(np.mean(np.abs(np.array(all_preds) - np.array(all_targets))))
    r, _       = pearsonr(all_preds, all_targets) if len(all_preds) > 2 else (0.0, None)
    avg_comps  = {k: v / n for k, v in comp_accum.items()}

    return avg_loss, mae, float(r), avg_comps



# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    train_model(
        csv_path                ="../.datasets/reviews_labeled.csv",
        model_name              = "microsoft/deberta-v3-large",
        epochs                  = 100,
        batch_size              = 32,
        lr                      = 2e-7,
        accumulation_steps      = 2,
        val_split               = 0.2,
        early_stopping_patience = 15,
        checkpoint_dir          ="../.weights/v4",
        load_model              = None,
        # Loss weights
        alpha    = 0.50,    # distillation
        beta     = 0.15,    # factuality
        gamma    = 0.20,    # geometric bound
        delta    = 0.15,    # lexical features
        margin_w = 0.10,    # ranking loss (set to 0.0 to disable)
        margin   = 0.08,    # minimum threshold between pairs for ranking loss
    )
