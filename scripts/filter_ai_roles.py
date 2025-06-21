import json
import os
import re
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any

# --- Config ---
RAW_FILE = "data/raw/10000_random_canadian_profiles.txt"
SAVE_PATH = "data/interim/profiles_ai_ml.parquet"
DEBUG = True  # Set to False to disable sample outputs

# --- Expanded Keywords --- (Now includes common variations)
TOPIC_KEYWORDS = {
    # Base terms
    "machine learning", "ml", "deep learning", "data science",
    "computer vision", "cv", "natural language processing", "nlp",
    "robotics", "reinforcement learning", "neural network",
    "llm", "large language model", "transformer",
    "artificial intelligence", "ai",
    
    # Tools/Frameworks (often indicate real AI work)
    "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
    "huggingface", "transformers", "openai", "langchain",
    
    # Techniques
    "supervised learning", "unsupervised learning", "generative ai",
    "cnn", "rnn", "lstm", "attention", "gan", "diffusion",
    
    # Applications
    "recommendation system", "time series", "object detection",
    "speech recognition", "sentiment analysis"
}

# --- Helper Functions ---
def normalize_text(text: str) -> str:
    """Lowercase and clean text for matching"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)  # Remove punctuation but keep hyphens
    return " ".join(text.split())  # Remove extra whitespace

def contains_ai_keywords(text: str) -> bool:
    """Check if text contains >=2 distinct AI keywords (flexible matching)"""
    if not text:
        return False
    
    text = normalize_text(text)
    found_keywords = set()
    
    # Check for multi-word phrases first
    for phrase in sorted(TOPIC_KEYWORDS, key=len, reverse=True):
        if len(phrase.split()) > 1 and phrase in text:
            found_keywords.add(phrase)
    
    # Check for single-word matches (avoid substrings)
    words = set(text.split())
    for word in TOPIC_KEYWORDS:
        if len(word.split()) == 1 and word in words:
            found_keywords.add(word)
    
    return len(found_keywords) >= 2  # Require at least 2 distinct matches

def is_ai_profile(profile: Dict[str, Any]) -> bool:
    """Determine if profile is AI-relevant with flexible checks"""
    # Safely extract all possible text fields
    text_parts = []
    for field in ["headline", "summary", "occupation", "title", "description"]:
        if field in profile and profile[field]:
            text_parts.append(str(profile[field]))
    
    # Check experiences if they exist
    for exp in profile.get("experiences", [])[:3]:  # Only check first 3 experiences
        for field in ["title", "description", "position"]:
            if field in exp and exp[field]:
                text_parts.append(str(exp[field]))
    
    # Check skills if they exist
    if "skills" in profile and profile["skills"]:
        if isinstance(profile["skills"], list):
            text_parts.append(" ".join(profile["skills"]))
        else:
            text_parts.append(str(profile["skills"]))
    
    combined_text = " ".join(text_parts)
    return contains_ai_keywords(combined_text)

# --- Main Pipeline ---
def load_profiles(file_path: str) -> pd.DataFrame:
    """Load and validate raw profiles"""
    profiles = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading profiles"):
            try:
                profile = json.loads(line.strip())
                if isinstance(profile, dict):  # Basic validation
                    profiles.append(profile)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    
    if DEBUG and profiles:
        print(f"\nSample raw profile (first of {len(profiles)}):")
        print(json.dumps(profiles[0], indent=2)[:500] + "...")
    
    return pd.DataFrame(profiles)

def filter_ai_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Apply AI/ML filtering with progress tracking"""
    ai_profiles = []
    discarded_samples = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering"):
        profile = row.to_dict()
        if is_ai_profile(profile):
            ai_profiles.append(profile)
        elif DEBUG and len(discarded_samples) < 3:
            discarded_samples.append(profile)
    
    if DEBUG and discarded_samples:
        print("\nSample discarded profiles:")
        for i, p in enumerate(discarded_samples[:3], 1):
            print(f"\nDiscarded #{i}:")
            print("Headline:", p.get("headline", ""))
            print("Summary:", (p.get("summary", "")[:200] + "...") if p.get("summary") else "")
    
    return pd.DataFrame(ai_profiles)

def main():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    print("Loading raw profiles...")
    df_raw = load_profiles(RAW_FILE)
    
    print("\nFiltering for AI/ML profiles...")
    df_ai = filter_ai_profiles(df_raw)
    
    print(f"\nFound {len(df_ai)} AI/ML profiles ({(len(df_ai)/len(df_raw))*100:.1f}%)")
    
    if not df_ai.empty:
        df_ai.to_parquet(SAVE_PATH, index=False)
        print(f"Saved to {SAVE_PATH}")
        
        if DEBUG:
            print("\nSample AI/ML profiles:")
            for i in range(min(3, len(df_ai))):
                print(f"\nProfile #{i+1}:")
                print("Headline:", df_ai.iloc[i].get("headline", ""))
                print("Skills:", df_ai.iloc[i].get("skills", "")[:200] + "...")
    else:
        print("Warning: No AI/ML profiles found!")

if __name__ == "__main__":
    main()