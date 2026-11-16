import os
import glob
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

def run_scraper():
    api = KaggleApi()
    api.authenticate()

    print("Hunting for new public notebooks...")
    kernels = api.kernels_list(competition="squares-in-squares", sort_by="hotness")
    
    os.makedirs("downloads", exist_ok=True)

    for k in kernels[:5]:
        print(f"Downloading outputs from: {k.ref}")
        try:
            api.kernels_output(k.ref, path="downloads")
        except Exception as e:
            print(f"Skipping {k.ref}: {e}")

    csv_files = glob.glob("downloads/**/*.csv", recursive=True)
    if not csv_files:
        print("No CSVs found.")
        return

    best_csv = max(csv_files, key=os.path.getsize)
    print(f"Selecting {best_csv} as the latest payload.")

    shutil.copy(best_csv, "latest_public.csv")
    print("Successfully updated latest_public.csv!")

if __name__ == "__main__":
    run_scraper()
