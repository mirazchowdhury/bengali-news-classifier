from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.ml.train_bert import NewsBertClassifier

app = Flask(__name__)

# Model Setup
MODEL_PATH = "models/best_news_bert/pytorch_model.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("models/best_news_bert")
model = NewsBertClassifier("sagorsarker/bangla-bert-base")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    news_title = ""
    news_content = ""
    confidence = 0
    probabilities = [0, 0]  # [Real %, Fake %]

    if request.method == "POST":
        news_title = request.form.get("news_title", "").strip()
        news_content = request.form.get("news_content", "").strip()
        combined_text = f"শিরোনাম: {news_title} খবর: {news_content}"

        if news_title or news_content:
            inputs = tokenizer(combined_text, max_length=256, padding='max_length', truncation=True,
                               return_tensors='pt')

            with torch.no_grad():
                outputs = model(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))
                probs = F.softmax(outputs, dim=1)  # Probability calculate kora

                label = torch.argmax(probs, dim=1).item()
                prediction = "Fake ❌" if label == 1 else "Real ✅"

                probabilities = [round(p * 100, 2) for p in probs[0].cpu().numpy().tolist()]
                confidence = probabilities[label]

    return render_template("index.html",
                           result=prediction,
                           news_title=news_title,
                           news_content=news_content,
                           confidence=confidence,
                           probs=probabilities)


if __name__ == "__main__":
    app.run(debug=True)