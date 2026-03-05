import os
import subprocess
import sys


def run_script(script_path):
    print(f"\n🚀 Executing: {script_path}")

    # Current environment er path set kora
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()  # Project root-ke path hishebe set kora

    try:
        # env=env parameter-ti add kora hoyeche
        result = subprocess.run([sys.executable, script_path], check=True, env=env)
        if result.returncode == 0:
            print(f"✅ Successfully finished: {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error occurred in {script_path}: {e}")
        sys.exit(1)


def main():
    # 1. Project directory ensure kora
    folders = ['data', 'models', 'models/best_news_bert']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")

    # 2. Step-by-step Pipeline Execution

    # Step A: Data Scraping (Real News)
    run_script("src/scraper/prothom_alo.py")
    # Step B: Data Scraping (Fake News)
    run_script("src/scraper/rumor_scanner.py")
    # Step C: Merging and Pre-processing
    run_script("src/processor/merged_dataset.py")
    # Step D: Training (Terminal output progress bar shoho dekhabe)
    #run_script("src/ml/train_bert.py")
    #run_script("src/scraper/ittefaq_tester.py")  # Scrap Ittefaq
    #run_script("src/ml/ittefaq_eval.py")  # Evaluate

    print("\n" + "🎉" * 10)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("Now you can run the dashboard using: python src/app/dashboard.py")
    print("🎉" * 10)


if __name__ == "__main__":
    main()