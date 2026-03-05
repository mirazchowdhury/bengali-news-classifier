import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import KFold


# --- 1. Custom Model Structure ---
class NewsBertClassifier(nn.Module):
    def __init__(self, model_name):
        super(NewsBertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        output = self.drop(pooled_output)
        return self.out(output)


# --- 2. Dataset Loader (Combined Title & Body) ---
class BanglaNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer(
            str(self.texts[item]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }


# --- 3. Validation Function ---
def evaluate(model, loader, device, loss_fn):
    model.eval()
    losses = []
    correct_predictions = 0
    total_elements = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            preds = torch.argmax(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_elements += labels.shape[0]
    return np.mean(losses), correct_predictions.double() / total_elements


# --- 4. Main K-Fold Training Loop ---
def run_kfold_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")

    df = pd.read_csv("data/final_news_dataset_cleaned.csv")
    df['label_num'] = df['Label'].map({'Real': 0, 'Fake': 1})

    # Combined column na thakle manual merge logic (safety)
    if 'Combined_Text' not in df.columns:
        df['Combined_Text'] = "শিরোনাম: " + df['Clean_Title'].fillna("") + " খবর: " + df['Clean_Body'].fillna("")

    tokenizer = BertTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
    kf = KFold(n_splits=5, shuffle=True, random_state=30)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n{'=' * 20} Fold {fold + 1} / 5 {'=' * 20}")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # Input e 'Combined_Text' pathano hocche
        train_ds = BanglaNewsDataset(train_df['Combined_Text'].tolist(), train_df['label_num'].tolist(), tokenizer,
                                     max_len=256)
        val_ds = BanglaNewsDataset(val_df['Combined_Text'].tolist(), val_df['label_num'].tolist(), tokenizer,
                                   max_len=256)

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

        model = NewsBertClassifier("sagorsarker/bangla-bert-base").to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        loss_fn = nn.CrossEntropyLoss().to(device)

        best_val_acc = 0

        for epoch in range(3):
            model.train()
            loop = tqdm(train_loader, leave=True)
            for batch in loop:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_description(f"Epoch {epoch + 1} Loss: {loss.item():.4f}")

            val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)
            print(f"Epoch {epoch + 1} Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if not os.path.exists("models/best_news_bert"): os.makedirs("models/best_news_bert")
                torch.save(model.state_dict(), "models/best_news_bert/pytorch_model.bin")
                tokenizer.save_pretrained("models/best_news_bert")

        fold_results.append(best_val_acc.item())

    print(f"\n📊 Final K-Fold Average Accuracy: {np.mean(fold_results):.4f}")


if __name__ == "__main__":
    run_kfold_training()