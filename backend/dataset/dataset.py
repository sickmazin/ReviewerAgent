import os
from typing import Optional
import numpy as np
import pandas as pd
import spacy
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from backend.ml_models.insightfulness_nn import InsightReviewScorer


# ============================================================
#  spaCy — loading with fallback
# ============================================================

def _load_spacy() -> spacy.language.Language:
    """
    Load en_core_web_sm and it_core_news_sm (Italian) is for fallback.
    Parser and NER are disabled for speed (using only the POS tagger).
    """
    for name in ["en_core_web_sm", "it_core_news_sm"]:
        try:
            return spacy.load(name, disable=["ner", "parser"])
        except OSError:
            continue
    raise RuntimeError("No spaCy model found. Install with: python -m spacy download en_core_web_sm or it_core_news_sm")


nlp = _load_spacy()


# ============================================================
#  Offline lexical features
# ============================================================

def extract_lexical_features(text: str) -> list:
    """
    Extracts 4 lexical features normalized in [0, 1]:
      [0] noun_ratio    — (NOUN + PROPN) / total
      [1] verb_ratio    — VERB / total
      [2] adj_ratio     — ADJ / total
      [3] entity_density— named entities / total

    These features are used as targets for the lexical_head during training,
    forcing the encoder to be sensitive to linguistic properties
    without requiring spaCy at runtime during inference.
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
    Ratio (NOUN + PROPN + VERB) / (ADJ + 2) as a factuality proxy.
    Value in [0, 1].
    """
    doc   = nlp(str(text))
    nouns = sum(1 for t in doc if t.pos_ in {"NOUN", "PROPN"})
    verbs = sum(1 for t in doc if t.pos_ == "VERB")
    adjs  = sum(1 for t in doc if t.pos_ == "ADJ")
    return float(min((nouns + verbs) / (adjs + 2.0), 1.0))


def preprocess_texts(texts: list, pbar: Optional[tqdm] = None) -> tuple[list, list]:
    """
    Calculates factuality_signal and lexical_features in parallel for a list of texts.
    Returns (factuality_signals: list[float], lexical_features: list[list[float]]).
    """
    num_proc = 1 if os.name == "nt" else min(os.cpu_count() or 1, 4)
    print(f"Preprocessing spaCy on {num_proc} cores...")

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
        print(f"[WARNING] Sequential fallback: {exc}")
        for t in tqdm(texts, desc="Preprocessing"):
            factuality_signals.append(extract_factuality_signal(t))
            lexical_features.append(extract_lexical_features(t))

    return factuality_signals, lexical_features


# ============================================================
#  Dataset
# ============================================================

N_LEXICAL_FEATS = 4     # must match n_lexical_feats in InsightReviewScorer

# ============================================================
#  Utilities
# ============================================================

def _get_embedding(text: str, model: InsightReviewScorer, tokenizer, device: str) -> torch.Tensor:
    """Pooled embedding for a single text, on CPU in float32."""
    # Apply prompt template for consistency with training/inference
    text_with_prompt = f"The score of the following review is [MASK]: {text}"

    inputs = tokenizer(
        text_with_prompt, return_tensors="pt", truncation=True, padding=True, max_length=512
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
    Calculates (or loads from cache) boundary embeddings for each category.
    Bounds are saved to disk: they are not recalculated on every run.
    Warning: if the model changes significantly, delete the cache file.
    """
    if os.path.exists(cache_path):
        print(f"Category bounds from cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    print("Calculating category bounds...")
    model.eval()
    bounds = {}
    for cat in tqdm(df["category"].unique(), desc="Categories"):
        cat_df    = df[df["category"] == cat]
        idx_worst = cat_df["insight_score"].idxmin()
        idx_best  = cat_df["insight_score"].idxmax()
        bounds[cat] = {
            "worst": _get_embedding(df.loc[idx_worst, "text"], model, tokenizer, device),
            "best":  _get_embedding(df.loc[idx_best,  "text"], model, tokenizer, device),
        }

    torch.save(bounds, cache_path)
    print(f"Category bounds saved to {cache_path}")
    return bounds


class ReviewDataset(Dataset):
    """
    PyTorch Dataset for InsightReviewScorer.

    Required columns in DataFrame:
      text, insight_score, category, factuality_signal,
      lex_noun, lex_verb, lex_adj, lex_ent
    """

    def __init__(
            self,
            tokenizer,
            model,
            device,
            bounds_cache,
            max_length: int = 512,
            chunk_size: int = 5000,
            cache_path: str = "../.datasets/tokenized_cache.pt",
            csv_path: str = "../.datasets/reviews_labeled.csv"
    ):
        self.df = self._preprocess_dataset(csv_path)

        if os.path.exists(cache_path):
            print(f"Loading tokenization from cache: {cache_path}")
            cached_data = torch.load(cache_path, weights_only=False)
            self.input_ids = cached_data["input_ids"]
            self.attention_mask = cached_data["attention_mask"]
        else:
            print(f"Tokenizing {len(self.df)} texts in chunks of {chunk_size}...")
            self._tokenize(tokenizer, chunk_size,  max_length, cache_path)

        # IMPORTANT: divide by 100 to align with the model's [0, 1] scale (Sigmoid)
        self.scores         = torch.tensor(self.df["insight_score"].values / 100.0, dtype=torch.float32)
        self.factuality     = torch.tensor(self.df["factuality_signal"].values, dtype=torch.float32)
        self.lexical        = torch.tensor(self.df[["lex_noun", "lex_verb", "lex_adj", "lex_ent"]].values, dtype=torch.float32)
        self.categories     = self.df["category"].values.tolist()
        self.bounds         = _compute_category_bounds(self.df, model, tokenizer, device, bounds_cache)

    def _tokenize(self, tokenizer, chunk_size,  max_length, cache_path):
        """
        Tokenizes texts in chunks to avoid RAM saturation.
        """
        all_texts = [f"The score of the following review is [MASK]: {t}" for t in self.df["text"].astype(str).tolist()]

        input_ids_list = []
        attention_mask_list = []

        # Chunk-based tokenization to prevent RAM saturation and show progress
        for i in tqdm(range(0, len(all_texts), chunk_size), desc="Tokenization"):
            chunk = all_texts[i : i + chunk_size]
            encodings = tokenizer(
                chunk,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids_list.append(encodings["input_ids"])
            attention_mask_list.append(encodings["attention_mask"])

        self.input_ids      = torch.cat(input_ids_list, dim=0)
        self.attention_mask = torch.cat(attention_mask_list, dim=0)

        # Save to disk for future runs
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask
        }, cache_path)
        print(f"Tokenization saved to {cache_path}")

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        cat    = self.categories[idx]
        bounds = self.bounds[cat]

        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "target_score":   self.scores[idx],
            "target_fact":    self.factuality[idx],
            "target_lexical": self.lexical[idx],
            "emb_worst":      bounds["worst"],
            "emb_best":       bounds["best"],
        }

    def _preprocess_dataset(self, csv_path):
        processed_csv = csv_path.replace(".csv", "_processed.csv")

        if os.path.exists(processed_csv):
            print(f"Preprocessed dataset found: {processed_csv}")
            df = pd.read_csv(processed_csv)
        else:
            df = pd.read_csv(csv_path)
            df = df.dropna(subset=["text", "insight_score"])
            df["text"] = df["text"].astype(str).str.strip()
            df = df[df["text"] != ""].reset_index(drop=True)

            print("Calculating spaCy features offline...")
            with tqdm(total=len(df), desc="spaCy") as pbar:
                facts, lexicals = preprocess_texts(df["text"].tolist(), pbar)

            df["factuality_signal"] = facts
            lex_arr = np.array(lexicals, dtype=np.float32)
            df["lex_noun"] = lex_arr[:, 0]
            df["lex_verb"] = lex_arr[:, 1]
            df["lex_adj"]  = lex_arr[:, 2]
            df["lex_ent"]  = lex_arr[:, 3]

            df.to_csv(processed_csv, index=False)
            print(f"Dataset saved to {processed_csv}")

        required_cols = ["text", "insight_score", "category",
            "factuality_signal", "lex_noun", "lex_verb", "lex_adj", "lex_ent"]
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        print(f"Dataset: {len(df)} rows, {df['category'].nunique()} categories")
        return df


if __name__ == "__main__":
    import os
    import pandas as pd

    # Default path used in ReviewDataset (relative to backend/dataset/ or root)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", ".datasets", "reviews_labeled.csv")

    if os.path.exists(csv_path):
        print(f"Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)

        def group_category(cat):
            cat_str = str(cat).lower()
            if cat_str in ["bnb", "restaurant"]:
                return cat_str
            return "amazon"

        df["grouped_category"] = df["category"].apply(group_category)
        counts = df["grouped_category"].value_counts()



        print("\n" + "="*40)
        print(f"{'CATEGORY':<20} | {'COUNT':<10}")
        print("-" * 40)
        for cat, count in counts.items():
            print(f"{cat:<20} | {count:<10}")
        print("="*40)
        print(f"{'TOTAL':<20} | {len(df):<10}")
        print("="*40)
    else:
        print(f"Dataset not found at: {csv_path}")
