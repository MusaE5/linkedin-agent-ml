import json
import os
import pandas as pd
import re
from tqdm import tqdm

# === Config ===
RAW_FILE = "data/raw/10000_random_canadian_profiles.txt"
SAVE_PATH = "data/interim/profiles_ai_ml.parquet"

# === Define AI/ML keyword pattern ===
pattern = re.compile(
    r"(AI|machine learning|ML|deep learning|data science|computer vision|NLP|natural language|robotics|reinforcement learning|neural network)",
    re.IGNORECASE,
)

def is_ai_related(profile):
    # Combine headline, summary, occupation, experiences, skills (if they exist)
    text_parts = []

    if profile.get("headline"):
        text_parts.append(profile["headline"])
    if profile.get("summary"):
        text_parts.append(profile["summary"])
    if profile.get("occupation"):
        text_parts.append(profile["occupation"])
    
    # Extract job titles from experiences
    for exp in profile.get("experiences", []):
        if "title" in exp and exp["title"]:
            text_parts.append(exp["title"])
        if "description" in exp and exp["description"]:
            text_parts.append(exp["description"])
    
    combined_text = " ".join(text_parts)
    return bool(pattern.search(combined_text))

def main():
    filtered_profiles = []

    with open(RAW_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scanning profiles"):
            try:
                profile = json.loads(line)
                if is_ai_related(profile):
                    filtered_profiles.append(profile)
            except json.JSONDecodeError:
                continue  # skip invalid lines

    print(f" Found {len(filtered_profiles)} AI/ML-related profiles")

    # Save to .parquet for fast access
    df = pd.DataFrame(filtered_profiles)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    df.to_parquet(SAVE_PATH, index=False)
    print(f" Saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
