# 🔍 Reviewer Agent: Hybrid Review Analysis System

Reviewer Agent is an advanced platform for automatic and intelligent review analysis. It combines the best of three worlds: **Fine Tuning (Transformer)** for qualitative scoring, **RAG (Retrieval-Augmented Generation)** for regulatory compliance, and **Structured LLMs** for grammatical and stylistic analysis.

The system is designed to help moderators and users evaluate the informational quality (insightfulness), compliance with platform-specific guidelines (Amazon, eBay, Google, etc.), and formal correctness of the text.

---

## 🏗️ System Architecture

The backend is orchestrated by a three-pillar architecture that processes each review in parallel:

### 1. The Neural Core: [`InsightReviewScorer`](backend/ml_models/insightfulness_nn.py)
A model based on **DeBERTaV3-Large** (state-of-the-art Transformer architecture) trained with a multi-task strategy to predict an "informativeness" score.

* **Attention Pooling**: Unlike classic models that use only the special `[CLS]` token, this system implements a scalar attention mechanism on every single token. This allows the model to dynamically weight the most relevant parts of a review (e.g. technical details vs filler), making it extremely effective on long texts.
* **Multi-Head Architecture** ([`insightfulness_nn.py`](backend/ml_models/insightfulness_nn.py)):
    * **Score Head**: 0-100 regression for the insightfulness score (distilled from an LLM "Teacher", in our case **Gemma3:27b**).
    * **Factuality Head**: Predicts the density of concrete information (nouns, verbs) versus vague adjectives.
    * **Lexical Head**: Internally learns linguistic properties (entity density, verbs, nouns) through spaCy supervision during training, with no runtime dependencies.
* **Composite Loss**: The model is trained by minimizing a loss that combines:
    * `MSE` for LLM distillation.
    * `Factuality Loss` and `Lexical Loss` for spaCy feature correction.
    * `Geometric Bound Loss`: Aligns the latent embedding space between the "worst" and "best" cases of each category.
    * `Margin Ranking Loss`: Ensures that better reviews always have higher scores than worse ones.

### 2. The RAG System: [`ReviewRAGSystem`](backend/ml_models/rag_and_compliance/rag_system.py)
An intelligent retrieval system that dynamically selects the correct platform rules.

* **Guideline Retriever** ([`guideline_retriever.py`](backend/ml_models/rag_and_compliance/guideline_retriever.py)): Uses a two-level strategy (Keyword Match + Semantic Embedding) to find the relevant policy document in a local database.
* **Compliance Checker** ([`compliance_checker.py`](backend/ml_models/rag_and_compliance/compliance_checker.py)):
    * **Heuristic Check**: Fast filter via Regex to identify privacy violations (phones, emails), external links, spam, and excessive uppercase usage.
    * **Contextual Check**: Uses a local LLM to interpret whether the text violates the platform's rules.

### 3. Grammar and Highlight Analysis
During the previously described checks, an LLM performs a linguistic analysis that identifies objective errors and suggests stylistic improvements.

* **Surgical Highlighting**: The system returns precise character indices (`start`, `end`) for each issue, allowing the frontend to highlight the text interactively.
* **Hallucination Filter**: A post-processing layer verifies that the errors flagged by the LLM actually exist in the original text and corrects any wrong indices.

---

## 📊 Dataset Composition

The dataset used for fine-tuning DeBERTa was built by integrating three heterogeneous review sources, selected to cover different domains and increase linguistic and semantic variability. The final composition is as follows:

| Category | Number of Reviews |
|---|---|
| BnB | 346,354 |
| Amazon | 133,295 |
| Restaurant | 35,128 |
| **Total** | **514,777** |

### Dataset Characteristics

* **Review Length**: Most reviews are under 150 words long. This is computationally relevant, as the text length determines the number of tokens processed by the encoder during training. The prevalence of relatively short reviews makes it possible to model the task with a fine-tuned Transformer encoder without necessarily resorting to an LLM with a much larger context window.
* **Insight Score Distribution**: The distribution is not normal but shows a multimodal trend. Multiple score concentrations are observable, with main peaks in the low-value area (around 15 and 35) and in the medium-high area (around 63 and 68-70). The frequency drop around the score of 55 suggests a separation between two macro-areas: one associated with low or medium-low informational reviews, and one associated with higher informational reviews.
* **Domain Dependency**: The box plot of scores by category highlights distinct behaviors:
    * **Amazon**: Distribution concentrated on generally low values, with a median around 28.
    * **Restaurant**: Similar to Amazon but slightly shifted towards higher values, with a median around 31.
    * **BnB**: Overall higher scores compared to the other two, with a median around 62, suggesting that stay-related reviews tend to contain more contextual and practical details.

<table width="100%" align="center">
  <tr>
    <td width="50%" align="center"><img src="backend/dataset/plots/insight_score_by_cat.png" alt="Insight score by category" width="100%"></td>
    <td width="50%" align="center"><img src="backend/dataset/plots/insight_score_dist.png" alt="Insight score distribution" width="100%"></td>
  </tr>
</table>
---

## 📈 Model Comparison

The project includes a benchmark system (`backend/comparison/`) that compares the local model with more powerful cloud models (Gemini 3.1 Flash Lite and GPT-OSS-120B):

* **Performance**: The local `Insightfulness` model is ~10x faster than Gemini 3.1 Flash (0.2s vs 2.1s per review).
* **Correlation**: Very high correlation (Pearson > 0.85) with the judgments of top-tier LLM models (GPT and Gemini).

The analysis shows that:
* The local DeBERTa Fine Tuned tends to be conservative compared to cloud models, with reviews concentrated around values between 20-40.
* The score correlation is strong with GPT-OSS (0.82), while it is more moderate with Gemini (0.62).
* The standard deviation of the score difference is higher with Gemini (20.76) than with GPT-OSS (15.58), indicating greater proximity of judgment with the latter.

<p align="center">
  <img src="backend/comparison/benchmark_plots.png" alt="Benchmark comparison plots" width="1080">
</p>

---

## 🧠 Training Pipeline & Results

### Training Overview

The training pipeline prepares the dataset, calculates auxiliary signals, builds category bounds, trains the model, and monitors performance on a validation set. The training is enriched with linguistic features and offline geometric references.

* **Tokenization**: Each review is tokenized with padding and truncation to a maximum length of 512 tokens.
* **Category Bounds**: For each category, the lowest and highest scoring reviews are identified. Their pooled embeddings become the worst and best references, used in the Geometric Bound Loss.
* **Optimizer**: AdamW with CosineAnnealingWarmRestarts scheduler (T_0 = steps_per_epoch, T_mult = 2). Gradient clipping with max norm 1.0.
* **Gradient Accumulation**: Supports accumulation to simulate larger effective batch sizes.
* **Early Stopping**: Stops training when validation loss does not improve for 10 epochs.

### Loss Weights

| Parameter | Value | Component |
|---|---|---|
| α | 0.50 | Distillation loss |
| β | 0.15 | Factuality loss |
| δ | 0.15 | Lexical feature loss |
| γ | 0.20 | Geometric bound loss |
| λ | 0.10 | Margin ranking loss |

### Results

The training lasted 117 epochs, preceded by a data enrichment phase managed via gemma3:27B:

* **Stall Phase**: The model learns very slowly with many oscillations in validation, indicating a high learning rate and/or a highly non-convex loss surface.
* **Drop Phase**: Precisely at epoch 93, the loss drops suddenly due to the transition to full fine-tuning.
* **Validation MAE**: Reached a minimum of 0.058 (5.8% deviation) on the normalized score.
* **Pearson R**: Reached a maximum of 0.93 after epoch 93.

<table width="100%" align="center">
  <tr>
    <td width="50%" align="center"><img src="backend/plot/1_loss_curves.png" alt="Training loss curves" width="100%"></td>
    <td width="50%" align="center"><img src="backend/plot/3_loss_components.png" alt="Training loss components" width="100%"></td>
  </tr>
</table>

### Reproducing the Training

1. **Labeling**: Use `dataset/generate_labels_ollama.py` to create "Silver Labels" using an LLM Teacher.
2. **Preprocessing**: The system automatically computes spaCy features and category bounds.
3. **Training**: Run `ml_models/training.py` (full fine-tuning) or `ml_models/training_frozen.py` (linear head on frozen backbone for VRAM saving).

---

## 🛠️ Tech Stack

* **Framework**: FastAPI (Python 3.10+)
* **Database**: PostgreSQL + SQLAlchemy
* **Machine Learning**: PyTorch, HuggingFace Transformers (DeBERTa-v3)
* **LLM Engine**: Ollama
* **Orchestration**: LangChain
* **NLP**: spaCy

---

## 🚀 Setup and Installation

### 1. Requirements
* Python 3.10 or higher.
* Ollama installed and running.
* PostgreSQL database.

### 2. Environment Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Database Configuration
The system waits for the DB to be ready before starting. Create a database named `reviewer_agent` and set the environment:
```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/reviewer_agent"
```

### 4. Models and Weights
Download the LLM model:
```bash
ollama pull gemma3:27b
```
Make sure the neural model weights are present in `backend/.weights/v7_frozen/epoch_117.pt`.

---

## 📁 Backend Structure

```
backend/
├── api.py                  # REST Endpoints and business logic
├── main.py                 # Entry point, DB init and model loading
├── schema.py               # DB models (SQLAlchemy) and Pydantic
├── ml_models/              # Core NLP
│   ├── Insightfulness_Model.py  # Main orchestrator (Facade)
│   ├── insightfulness_nn.py     # Neural Network architecture (PyTorch)
│   └── rag_and_compliance/      # RAG system and compliance logic
├── dataset/                # Data generation and labeling pipeline
├── comparison/             # Scripts for benchmark and LLM comparisons
└── .prompts/               # Engineered prompts for different analyses
```

---

## 🔌 Main API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/evaluate` | Full review analysis (Neural + RAG + LLM) |
| `GET` | `/chats` | Retrieve analysis session history |
| `GET` | `/chats/{id}/review` | Detail of reviews analyzed in a chat |
| `GET` | `/sites` | List of supported platforms (Amazon, eBay, etc.) |
| `GET` | `/model-info` | Technical model metadata for the frontend |

## Visualization

The following screenshot shows an example review analyzed through the frontend. It demonstrates the complete system flow: the user submits a review, the backend evaluates insightfulness, compliance, grammar, and style, and the frontend presents the resulting scores, explanations, and highlighted issues in a single interactive view.

<p align="center">
  <img src="docs/web_view.png" alt="Frontend visualization of an analyzed review" width="1080">
</p>

---

##### This project has been developed as a coursework assigned for the class "Information Retrieval and Natural Language Processing", Summer 2026 (Instructor: prof. [Andrea Tagarelli](https://mlnteam-unical.github.io/)) at the DIMES Department, University of Calabria, Italy

#### Developed with care and dedication by Mattia Corigliano and Paolo Costa.
