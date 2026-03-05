import torch
from transformers import BertTokenizer
import sys
import os

# Model architecture same rakhar jonno train_bert theke import kora bhalo
# Shudhu dashboard/classifier er jonno ekhane class-ti thaka dorkar
try:
    # Manual run ba external import-er jonno
    from src.ml.train_bert import NewsBertClassifier
except ImportError:
    # Direct ml folder theke run korle
    from train_bert import NewsBertClassifier


class NewsClassifier:
    def __init__(self, model_path="models/best_news_bert"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)

        # Load Model
        self.model = NewsBertClassifier("sagorsarker/bangla-bert-base")
        self.model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        # self.tokenizer.encode_plus er bodole self.tokenizer(...) use kora
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=256,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            # Model output calculation
            outputs = self.model(input_ids, attention_mask)
            prediction = torch.argmax(outputs, dim=1).item()

        return "Fake ❌" if prediction == 1 else "Real ✅"


if __name__ == "__main__":
    # Test korar jonno
    clf = NewsClassifier()
    test_text = "আপনার নিউজ টেক্সট এখানে দিন"
    print(f"Result: {clf.predict(test_text)}")