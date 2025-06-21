import pandas as pd
import random

# Load filtered AI/ML profiles
df = pd.read_parquet("data/interim/profiles_ai_ml.parquet")

# Sample 30 random profiles for manual audit
sample = df.sample(27, random_state=42)

for i, row in sample.iterrows():
    print(f"\n=== Profile {i+1} ===")
    print("Full Name:", row.get("full_name", "N/A"))
    print("Headline:", row.get("headline", ""))
    print("Summary:", row.get("summary", ""))
    print("Occupation:", row.get("occupation", ""))
    
    experiences = row.get("experiences", [])
    if isinstance(experiences, list):
        for exp in experiences[:2]:
            print(" - Job Title:", exp.get("title", ""))
            print("   Description:", exp.get("description", ""))
