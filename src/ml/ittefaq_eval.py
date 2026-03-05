import pandas as pd
import torch
from classifier import NewsClassifier  # Age likhe deya classifier logic
import os


def test_ittefaq_data():
    input_file = "data/ittefaq_real_news.csv"
    model_path = "models/best_news_bert"

    if not os.path.exists(input_file):
        print("Error: Ittefaq data file pawa jayni! Age scraper-ti run korun.")
        return

    # 1. Model & Classifier Load
    print("Model load hochhe...")
    clf = NewsClassifier(model_path=model_path)

    # 2. Data Load
    df = pd.read_csv(input_file)
    print(f"Total {len(df)} ti news test kora hobe...")

    results = []

    # 3. Prediction Loop
    for index, row in df.iterrows():
        text = str(row['Title']) + " " + str(row['Body'])
        prediction = clf.predict(text)
        results.append(prediction)

        # Terminal-e live prediction dekhano
        print(f"[{index + 1}] Title: {row['Title'][:50]}... -> Result: {prediction}")

    # 4. Result Save
    df['Model_Prediction'] = results
    output_file = "data/ittefaq_test_results.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    # Accuracy Report (Jehetu amra jani egulo Real news)
    real_count = results.count("Real ✅")
    accuracy = (real_count / len(results)) * 100
    print(f"\n--- Evaluation Report ---")
    print(f"Total Tested: {len(results)}")
    print(f"Correctly Identified (Real): {real_count}")
    print(f"Accuracy on Ittefaq: {accuracy:.2f}%")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    test_ittefaq_data()