from datasets import load_dataset
import pandas as pd
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

OUTPUT_DIR = Path("data/sample")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading MMLU...")

dataset = load_dataset(
    "cais/mmlu",
    "all",
    split="test"
)

print(f"Loaded {len(dataset)} MMLU questions")

# Sample 5 questions for our first test
indices = random.sample(range(len(dataset)), 5)

records = []

for idx in indices:
    item = dataset[idx]

    records.append({
        "question_id": f"mmlu_{idx}",
        "source": "MMLU",
        "subject": item["subject"],
        "question": item["question"],
        "options": item["choices"],
        "correct_answer": "ABCD"[item["answer"]],
    })

df = pd.DataFrame(records)

output_path = OUTPUT_DIR / "mmlu_test_5.csv"
df.to_csv(output_path, index=False)

print(df[["question_id", "subject", "correct_answer"]])
print(f"Saved to {output_path}")