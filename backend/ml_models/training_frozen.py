"""
training_frozen.py
==================
Variant of the training script with a frozen backbone (DeBERTa).
Trains only the AttentionPooling and the Heads (Score, Factuality, Lexical).

Useful for:
- Saving VRAM with DeBERTa-v3-large
- Speeding up training
- Pure feature extraction
"""

from training import train_model

if __name__ == "__main__":
    # Optimized configuration for "Freeze Backbone"
    train_model(
        csv_path                ="../.datasets/reviews_labeled.csv",
        model_name              = "microsoft/deberta-v3-large",
        epochs                  = 200,
        batch_size              = 1,     # Batch size can be increased as the encoder is frozen
        lr                      = 5e-6,     # Higher LR (1e-4 or 5e-4) as we only train the heads
        accumulation_steps      = 16,      # Less accumulation needed
        val_split               = 0.1,
        early_stopping_patience = 15,
        checkpoint_dir          ="../.weights/v7_frozen",
        load_model              = None,
        freeze_encoder          = True,   # <--- Key parameter
        # Loss weights (balanced)
        alpha    = 0.50,    # distillation
        beta     = 0.15,    # factuality
        gamma    = 0.20,    # geometric bound
        delta    = 0.15,    # lexical features
        margin_w = 0.0,     # ranking loss
        margin   = 0.08,    # minimum threshold between pairs for ranking loss
    )
