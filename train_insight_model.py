import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from tqdm import tqdm
from insightful_model import InsightScorer, HybridInsightSystem
import os

# Configurazione Training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

class ReviewDataset(Dataset):
    """
    Dataset per recensioni Amazon con generazione dinamica di Silver Labels.
    """
    def __init__(self, csv_path, tokenizer, max_length=512):
        self.df = pd.read_csv(csv_path)
        # Pulizia base: rimuove righe senza testo
        self.df = self.df.dropna(subset=['text'])
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)

    def _generate_silver_labels(self, row):
        """
        Legge lo score generato dall'LLM (distillazione) invece di usare euristiche.
        """
        insight_score = row.get('insight_score', 50.0) # Default se non presente
        
        # Etichetta Soggettività: Possiamo ancora derivarla dal testo o
        # (Opzionale) Chiedere anche questa all'LLM in fase di labeling
        text = str(row['text'])
        is_subjective = 1.0 if len(text.split()) < 20 else 0.2
        
        return torch.tensor(insight_score, dtype=torch.float), torch.tensor(is_subjective, dtype=torch.float)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])
        
        score_target, subj_target = self._generate_silver_labels(row)
        
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

def train_model(train_csv_path, model_save_path="insight_model_v1.pth"):
    # Inizializzazione Sistema
    system = HybridInsightSystem()
    tokenizer = system.tokenizer
    model = system.model.to(DEVICE)
    
    dataset = ReviewDataset(train_csv_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # Loss Functions
    mse_loss = nn.MSELoss() # Per lo score 0-100
    bce_loss = nn.BCELoss() # Per la soggettività 0-1
    
    model.train()
    print(f"Inizio addestramento su {DEVICE}...")
    
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
            # Nota: Ignoriamo 'aspects' per ora (richiederebbe label multi-class specifiche)
            score_pred, subj_pred, _, _ = model(input_ids, attention_mask)
            
            # Calcolo Loss Ibrida
            loss_score = mse_loss(score_pred.flatten(), score_target)
            loss_subj = bce_loss(subj_pred.flatten(), subj_target)
            
            # L_total = λ1 * L_score + λ2 * L_subj
            loss = loss_score + (loss_subj * 10) # Peso maggiore alla soggettività per stabilizzare
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} completata. Average Loss: {total_loss/len(dataloader):.4f}")
        
    # Salvataggio pesi
    torch.save(model.state_dict(), model_save_path)
    print(f"Modello salvato in {model_save_path}")

if __name__ == "__main__":
    # Esempio con il dataset Electronics già presente nel workspace
    TRAIN_PATH = "datasets/Amazon-Reviews-2023/benchmark/0core/last_out/Electronics.train.csv"
    
    if os.path.exists(TRAIN_PATH):
        # Usiamo solo un sottoinsieme per velocizzare l'esempio (opzionale)
        train_model(TRAIN_PATH)
    else:
        print(f"Errore: Dataset non trovato in {TRAIN_PATH}")
