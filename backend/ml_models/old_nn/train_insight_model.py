import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from tqdm import tqdm
from insightful_model import HybridInsightSystem
import os
import argparse


class ReviewDataset(Dataset):
    """
    Dataset per la distillazione della conoscenza (Knowledge Distillation) 
    dalle Silver Labels generate tramite LLM.
    """
    def __init__(self, csv_path, tokenizer, max_length=512):
        print(f"Loading dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        # Assicuriamo che le colonne necessarie esistano e non siano nulle
        self.df = self.df.dropna(subset=['text', 'insight_score'])
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)

    def _get_targets(self, row):
        """
        Estrae lo score dall'LLM e calcola una proxy di soggettività.
        """
        insight_score = float(row['insight_score'])
        text = str(row['text'])
        
        # Proxy soggettività: 1.0 (soggettivo/corto), 0.0 (oggettivo/dettagliato)
        # Inversa logaritmica della lunghezza come segnale debole di complessità
        is_subjective = 1.0 / (1.0 + np.log1p(len(text.split())))
        
        return torch.tensor(insight_score, dtype=torch.float), torch.tensor(is_subjective, dtype=torch.float)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])
        
        score_target, subj_target = self._get_targets(row)
        
        inputs = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'score_target': score_target,
            'subj_target': subj_target
        }

def train_model(train_csv_path, model_save_path="insight_model_v1.pth", base_model_path=None):
    # Creazione cartella per i checkpoint
    os.makedirs("../../.weights",exist_ok=True)
    
    # Inizializzazione Sistema (Carica base_model_path se fornito)
    system = HybridInsightSystem(model_path=base_model_path, device=DEVICE)
    tokenizer = system.tokenizer
    model = system.model
    
    dataset = ReviewDataset(train_csv_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Optimizer con Weight Decay
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    total_steps = len(dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Loss Functions
    mse_loss = nn.MSELoss() 
    bce_loss = nn.BCELoss() 
    
    model.train()
    print(f"Inizio fine-tuning su {DEVICE}...")
    if base_model_path:
        print(f"Partendo dai pesi di: {base_model_path}")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            score_target = batch['score_target'].to(DEVICE)
            subj_target = batch['subj_target'].to(DEVICE)
            
            # Forward Multi-task
            score_pred, subj_pred, _, _ = model(input_ids, attention_mask)
            
            # Bilanciamento dimensionale dei gradienti (MSE scale correction)
            loss_score = mse_loss(score_pred.flatten(), score_target) / 100.0
            loss_subj = bce_loss(subj_pred.flatten(), subj_target)
            
            loss = loss_score + loss_subj
            
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'mse': f"{loss_score.item()*100:.2f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completata. Average Loss: {avg_loss:.4f}")
        
        # Salvataggio checkpoint per epoca
        checkpoint_path = os.path.join("../../.weights",f"insight_model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint salvato: {checkpoint_path}")

# Configurazione Training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 24
EPOCHS = 100
LEARNING_RATE = 2e-5
TRAIN_PATH= "../../.datasets/reviews_labeled.csv"
MODEL_NAME=""

if __name__ == "__main__":

    if os.path.exists(TRAIN_PATH):
        train_model(
            train_csv_path=TRAIN_PATH,
            model_save_path=MODEL_NAME
        )
    else:
        print(f"Errore: Dataset non trovato in {TRAIN_PATH}. Esegui prima generate_labels_ollama.py.")
