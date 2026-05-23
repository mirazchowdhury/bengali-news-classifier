# Bengali News Classifier

A Bangla news classification project for detecting whether a Bengali news article is real or fake. The project collects real news from Prothom Alo, collects fake or rumor based news from Rumor Scanner, builds a balanced cleaned dataset, trains a Bangla BERT classifier, and serves predictions through a Flask based web dashboard.

## Project Overview

This repository is an end to end Bengali news classification pipeline. It includes web scraping, text cleaning, dataset preparation, transformer based model training, model inference, external newspaper testing, and a small web dashboard for manual prediction.

The system uses the `sagorsarker/bangla-bert-base` model as the base encoder. A custom PyTorch classification head is added on top of BERT to predict two labels:

```text
Real
Fake
```

## Repository Link

```text
https://github.com/mirazchowdhury/bengali-news-classifier
```

## Main Features

1. Scrapes real Bangla news from the Bangladesh section of Prothom Alo.
2. Scrapes fake or rumor related Bangla news from Rumor Scanner.
3. Supports incremental scraping by skipping previously collected URLs.
4. Cleans Bangla text by removing URLs, HTML tags, and non Bangla characters.
5. Creates a balanced dataset from real and fake news records.
6. Combines news title and body into one model input text.
7. Trains a Bangla BERT based binary classifier.
8. Uses 5 fold cross validation during model training.
9. Saves the best model checkpoint in the `models` directory.
10. Loads the trained model for live prediction.
11. Provides a Flask web dashboard for entering news title and body.
12. Returns Real or Fake prediction with probability based confidence.
13. Includes an Ittefaq real news evaluation script.
14. Includes saved data CSV files and saved model files.

## Current Repository Structure

```text
bengali-news-classifier/
    main.py
    requirements.txt
    command.txt
    .gitattributes

    data/
        final_news_dataset_cleaned.csv
        ittefaq_real_news.csv
        ittefaq_test_results.csv
        prothom_alo_bangladesh_real.csv
        rumor_data_fake.csv

    models/
        best_news_bert/
            pytorch_model.bin
            tokenizer.json
            tokenizer_config.json

    src/
        __init__.py

        app/
            dashboard.py
            templates/
                index.html

        ml/
            classifier.py
            ittefaq_eval.py
            train_bert.py

        processor/
            merged_dataset.py

        scraper/
            inquilab_tester.py
            prothom_alo.py
            rumor_scanner.py

    utils/
        __init__.py
        cleaner.py
```

## Folder Details

## data

This folder stores the scraped and processed CSV files.

| File | Purpose |
| :-- | :-- |
| `prothom_alo_bangladesh_real.csv` | Real Bangla news collected from Prothom Alo |
| `rumor_data_fake.csv` | Fake or rumor based Bangla news collected from Rumor Scanner |
| `final_news_dataset_cleaned.csv` | Balanced and cleaned final dataset used for training |
| `ittefaq_real_news.csv` | Ittefaq news used for external real news testing |
| `ittefaq_test_results.csv` | Prediction results produced by the Ittefaq evaluation script |

## models

This folder stores the trained model artifacts.

```text
models/best_news_bert/
    pytorch_model.bin
    tokenizer.json
    tokenizer_config.json
```

The Flask dashboard and classifier class load the trained model from this folder.

## src scraper

This folder contains web scraping scripts.

| Script | Purpose |
| :-- | :-- |
| `prothom_alo.py` | Scrapes real news from Prothom Alo Bangladesh section |
| `rumor_scanner.py` | Scrapes fake or rumor based news from Rumor Scanner |
| `inquilab_tester.py` | Scrapes Ittefaq news for external real news evaluation |

## src processor

This folder contains dataset preparation logic.

| Script | Purpose |
| :-- | :-- |
| `merged_dataset.py` | Reads real and fake CSV files, balances both classes, cleans text, creates `Combined_Text`, and saves the final dataset |

## src ml

This folder contains model training and inference logic.

| Script | Purpose |
| :-- | :-- |
| `train_bert.py` | Defines the Bangla BERT classifier and trains it with 5 fold cross validation |
| `classifier.py` | Loads the trained model and predicts Real or Fake for a given text |
| `ittefaq_eval.py` | Tests trained model on Ittefaq real news and saves prediction results |

## src app

This folder contains the Flask dashboard.

| Script | Purpose |
| :-- | :-- |
| `dashboard.py` | Loads the trained model, accepts title and body input, predicts Real or Fake, and renders the result |
| `templates/index.html` | HTML template for the prediction page |

## utils

This folder contains reusable helper code.

| Script | Purpose |
| :-- | :-- |
| `cleaner.py` | Cleans Bangla text by removing URLs, HTML tags, and non Bangla symbols |

## Full Pipeline

```text
Prothom Alo scraper
    Real news CSV

Rumor Scanner scraper
    Fake news CSV

Dataset processor
    Balanced cleaned final dataset

Bangla BERT trainer
    Best saved model

Classifier and dashboard
    Real or Fake prediction
```

## Workflow Description

## Step 1: Real News Scraping

The `src/scraper/prothom_alo.py` script uses Playwright to open the Prothom Alo Bangladesh section, collect news links, visit each article page, extract title, author, date, body text, URL, and label the record as `Real`.

Output file:

```text
data/prothom_alo_bangladesh_real.csv
```

## Step 2: Fake News Scraping

The `src/scraper/rumor_scanner.py` script uses Playwright to open Rumor Scanner, collect article links, visit each article page, extract article content, and label the record as `Fake`.

Output file:

```text
data/rumor_data_fake.csv
```

## Step 3: Dataset Cleaning and Merging

The `src/processor/merged_dataset.py` script reads both real and fake news CSV files. It takes the same number of rows from each class, cleans the Bangla title and body, combines them into one text column, shuffles the final dataset, and saves it as UTF 8 CSV.

Output file:

```text
data/final_news_dataset_cleaned.csv
```

Important final columns:

```text
Title
Body
Label
Clean_Title
Clean_Body
Combined_Text
```

The model uses `Combined_Text` as the main input.

## Step 4: Model Training

The `src/ml/train_bert.py` script trains a binary classifier using Bangla BERT.

Model base:

```text
sagorsarker/bangla-bert-base
```

Architecture summary:

1. Bangla BERT encoder.
2. Dropout layer with value `0.3`.
3. Linear classification layer with 2 output classes.
4. Cross entropy loss.
5. AdamW optimizer.
6. 5 fold cross validation.

Training settings used in the script:

| Setting | Value |
| :-- | :-- |
| Maximum token length | 256 |
| Batch size | 16 |
| Learning rate | 2e 5 |
| Epochs per fold | 3 |
| Number of folds | 5 |
| Label mapping | Real = 0, Fake = 1 |

Saved model path:

```text
models/best_news_bert/pytorch_model.bin
```

## Step 5: Model Inference

The `src/ml/classifier.py` script loads the saved tokenizer and model from:

```text
models/best_news_bert
```

It tokenizes user text, sends the input through the model, takes the class with the highest output score, and returns:

```text
Real ✅
Fake ❌
```

## Step 6: Flask Dashboard

The `src/app/dashboard.py` script starts a Flask web app. The dashboard accepts:

1. News title.
2. News body.

Then it combines both fields as:

```text
শিরোনাম: <title> খবর: <body>
```

The model predicts the label and computes softmax probability for both classes. The page shows the prediction, confidence, and class probabilities.

## Technology Stack

| Area | Technology |
| :-- | :-- |
| Language | Python |
| Scraping | Playwright |
| Data processing | Pandas |
| Text cleaning | Regex |
| Model framework | PyTorch |
| Transformer library | Hugging Face Transformers |
| Base model | Bangla BERT |
| Training utilities | scikit learn, tqdm |
| Dashboard | Flask |
| Visualization package | Plotly |
| Spreadsheet support | OpenPyXL |
| Optional app package | Streamlit |

## Installation

Clone the repository.

```bash
git clone https://github.com/mirazchowdhury/bengali-news-classifier.git
cd bengali-news-classifier
```

Create a virtual environment.

```bash
python -m venv .news_portal_scrapping
```

Activate the environment on Windows.

```bash
.news_portal_scrapping\Scripts\activate
```

Activate the environment on Linux or macOS.

```bash
source .news_portal_scrapping/bin/activate
```

Upgrade pip.

```bash
python -m pip install --upgrade pip
```

Install required packages.

```bash
pip install -r requirements.txt
```

Install Playwright browser binaries.

```bash
python -m playwright install chromium
```

For CUDA 11.8 based PyTorch installation, the command in `command.txt` uses:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Use the PyTorch command that matches your own GPU, CUDA version, and operating system.

## Requirements

The repository uses the following packages:

```text
playwright
pandas
beautifulsoup4
sentence-transformers
python-dotenv
asyncio
openpyxl
streamlit
flask
torch
transformers
scikit-learn
plotly
tqdm
datasets
```

Clean suggested `requirements.txt` format:

```text
playwright
pandas
beautifulsoup4
sentence-transformers
python-dotenv
openpyxl
streamlit
flask
torch
transformers
scikit-learn
plotly
tqdm
datasets
```

Note that `asyncio` is part of the Python standard library and usually does not need to be installed separately.

## How to Run the Full Data Pipeline

Run the root pipeline script.

```bash
python main.py
```

The current root pipeline performs these steps:

1. Creates required folders if missing.
2. Runs Prothom Alo scraper.
3. Runs Rumor Scanner scraper.
4. Runs dataset merger and cleaner.

At the end, it prints the command for starting the dashboard.

Important note:

In the current `main.py`, the model training line is commented out. To train from the root pipeline, uncomment this line:

```python
run_script("src/ml/train_bert.py")
```

Or run the training file manually.

## How to Run Each Step Manually

Run real news scraping.

```bash
python src/scraper/prothom_alo.py
```

Run fake news scraping.

```bash
python src/scraper/rumor_scanner.py
```

Merge and clean the dataset.

```bash
python src/processor/merged_dataset.py
```

Train the Bangla BERT classifier.

```bash
python src/ml/train_bert.py
```

Run Ittefaq external test.

```bash
python src/scraper/inquilab_tester.py
python src/ml/ittefaq_eval.py
```

Run the Flask dashboard.

```bash
python src/app/dashboard.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Dashboard Usage

1. Start the dashboard.
2. Open the browser at `http://127.0.0.1:5000`.
3. Enter a Bangla news title.
4. Enter a Bangla news body.
5. Submit the form.
6. View the predicted label, confidence score, and probability values.

## Prediction Output

The dashboard returns one of two labels:

```text
Real ✅
Fake ❌
```

It also returns the confidence value based on the predicted class probability.

Example logical output:

```text
Prediction: Real ✅
Confidence: 91.25 percent
Real probability: 91.25 percent
Fake probability: 8.75 percent
```

## Dataset Format

The final training CSV should contain at least these columns:

```text
Title
Body
Label
Clean_Title
Clean_Body
Combined_Text
```

Label values:

```text
Real
Fake
```

The training script internally maps labels as:

```text
Real = 0
Fake = 1
```

## Model Artifacts

The dashboard expects these files to exist:

```text
models/best_news_bert/pytorch_model.bin
models/best_news_bert/tokenizer.json
models/best_news_bert/tokenizer_config.json
```

If these files are missing, train the model again or copy the saved model artifacts into the correct folder.

## External Evaluation With Ittefaq

The repository includes an Ittefaq evaluation flow. The Ittefaq scraper saves real news into:

```text
data/ittefaq_real_news.csv
```

The evaluation script loads the trained classifier, predicts each Ittefaq article, writes predictions to:

```text
data/ittefaq_test_results.csv
```

It then prints the number of articles identified as Real and computes accuracy under the assumption that all Ittefaq records are real news.

## Common Errors and Fixes

## Playwright browser not found

Cause:

Chromium is not installed for Playwright.

Fix:

```bash
python -m playwright install chromium
```

## Scraped data files missing

Cause:

The dataset merger was run before the scrapers produced CSV files.

Fix:

Run both scrapers first.

```bash
python src/scraper/prothom_alo.py
python src/scraper/rumor_scanner.py
python src/processor/merged_dataset.py
```

## Model file not found

Cause:

`pytorch_model.bin` is missing from `models/best_news_bert`.

Fix:

Run the training script.

```bash
python src/ml/train_bert.py
```

## Tokenizer file not found

Cause:

Tokenizer files are missing from `models/best_news_bert`.

Fix:

Make sure the trained tokenizer files are saved in the same directory as the model.

## Import error while running scripts

Cause:

Python cannot find project modules.

Fix:

Run commands from the repository root. The root `main.py` sets `PYTHONPATH` to the project root when running child scripts.

## CUDA out of memory

Cause:

BERT training needs GPU memory.

Fix options:

1. Reduce batch size from 16 to 8 or 4.
2. Use CPU mode if GPU memory is too low.
3. Use a smaller transformer model.
4. Train on a cloud GPU.

## Browser opens during scraping

Cause:

The scraper uses `headless=False`.

Fix:

Change:

```python
browser = await p.chromium.launch(headless=False)
```

to:

```python
browser = await p.chromium.launch(headless=True)
```

## Current Limitations

1. The repository has no license file visible.
2. The model training step is commented out in `main.py`.
3. Some generated folders such as `__pycache__` are committed.
4. The `.idea` folder is committed and should be ignored.
5. The scraper depends on current website structure, so CSS selectors may break if the source sites change.
6. Scraping is performed with visible browser mode by default.
7. No API endpoint is currently exposed for prediction.
8. The dashboard is Flask based and intended for local use.
9. No automated unit tests are included.
10. Class balance is created by down sampling to the smaller class size, which may discard usable data.
11. The model predicts binary labels only and does not explain why an article was predicted as real or fake.
12. The current project does not include data licensing notes for scraped articles.

## Recommended Improvements

1. Add a license file.
2. Add `.gitignore` entries for `.idea`, `__pycache__`, virtual environments, and temporary files.
3. Add `.env` support for configurable scraping values.
4. Add a FastAPI prediction endpoint.
5. Add model evaluation metrics such as precision, recall, F1 score, and confusion matrix.
6. Add test split evaluation in addition to cross validation.
7. Add saved training logs.
8. Add explainability support using attention visualization or token importance.
9. Add rate limiting and polite crawling delay for scrapers.
10. Add dataset source and usage notes.
11. Add a model card for the trained Bangla BERT model.
12. Add Docker support.
13. Add GitHub Actions for syntax checking.
14. Add a small sample input and sample output section.
15. Add batch prediction support for CSV files.
16. Add support for more real news sources.
17. Add support for more fact checking sources.

## Suggested Git Ignore

```text
.venv/
.news_portal_scrapping/
__pycache__/
*.pyc
.idea/
.env
*.log
data/raw/
outputs/
```

## Suggested Future API Design

A future API could expose prediction through this endpoint:

```text
POST /predict
```

Request body:

```json
{
    "title": "বাংলা খবরের শিরোনাম",
    "body": "বাংলা খবরের বিস্তারিত লেখা"
}
```

Response body:

```json
{
    "label": "Real",
    "confidence": 91.25,
    "probabilities": {
        "Real": 91.25,
        "Fake": 8.75
    }
}
```

## Ethical and Legal Notes

This project works with news content and misinformation related classification. Before using it publicly, consider the following points:

1. Respect website terms when scraping data.
2. Avoid overloading source websites.
3. Keep source links and labels traceable.
4. Do not treat model output as final fact checking.
5. Use human review for sensitive claims.
6. Clearly state model limitations to users.
7. Keep data collection and publication practices transparent.

## License

No license file was visible in the repository when this README was prepared. Add a license before public reuse or commercial use.

## Author

Miraz Uddin Chowdhury

Repository:

```text
https://github.com/mirazchowdhury/bengali-news-classifier
```
