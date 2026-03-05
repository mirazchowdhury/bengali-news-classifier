import pandas as pd
import os
from utils.cleaner import clean_bangla_text


# Note: BERT use korle TF-IDF feature engineering mandatory noy, tobe save rakha bhalo

def process_and_merge():
    real_path = "data/prothom_alo_bangladesh_real.csv"
    fake_path = "data/rumor_data_fake.csv"
    output_path = "data/final_news_dataset_cleaned.csv"

    if not os.path.exists(real_path) or not os.path.exists(fake_path):
        print("Error: Scraped data files missing!")
        return

    df_real = pd.read_csv(real_path)
    df_fake = pd.read_csv(fake_path)

    min_size = min(len(df_real), len(df_fake))
    df_real_sampled = df_real[['Title', 'Body', 'Label']].sample(n=min_size, random_state=42)
    df_fake_sampled = df_fake[['Title', 'Body', 'Label']].sample(n=min_size, random_state=42)

    final_df = pd.concat([df_real_sampled, df_fake_sampled], ignore_index=True)

    print("Cleaning Bangla Text...")
    final_df['Body'] = final_df['Body'].fillna("")
    final_df['Title'] = final_df['Title'].fillna("")

    final_df['Clean_Body'] = final_df['Body'].apply(clean_bangla_text)
    final_df['Clean_Title'] = final_df['Title'].apply(clean_bangla_text)

    # Title + Body merge kora (Training er jonno main input)
    final_df['Combined_Text'] = "শিরোনাম: " + final_df['Clean_Title'] + " খবর: " + final_df['Clean_Body']

    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"SUCCESS! Balanced dataset saved with Combined_Text. Rows: {len(final_df)}")


if __name__ == "__main__":
    process_and_merge()