from datasets import load_dataset
import pandas as pd
import random
import json
from collections import defaultdict
from pathlib import Path

SEED = 42
SAMPLE_SIZE = 40
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

# Group question indices by subject
subjects = defaultdict(list)

for idx, item in enumerate(dataset):
    subjects[item["subject"]].append(idx)

# Randomly select 40 different subjects
rng = random.Random(SEED)
selected_subjects = rng.sample(sorted(subjects.keys()), SAMPLE_SIZE)

indices = [
    rng.choice(subjects[subject])
    for subject in selected_subjects
]

records = []

for idx in indices:
    item = dataset[idx]

    records.append({
        "question_id": f"mmlu_{idx}",
        "source": "MMLU",
        "subject": item["subject"],
        "question": item["question"],
        "options": json.dumps(dict(zip("ABCD", item["choices"]))),
        "correct_answer": "ABCD"[int(item["answer"])],
    })
df = pd.DataFrame(records)

output_path = OUTPUT_DIR / "mmlu_40.csv"
print("Number of selected questions:", len(df))
print("Number of sampled indices:", len(indices))
assert len(df) == 40
assert df["question_id"].is_unique
assert df["correct_answer"].isin(list("ABCD")).all()
assert df["question"].notna().all()

df.to_csv(output_path, index=False)

print(df[["question_id", "subject", "correct_answer"]])
print(f"Saved to {output_path}")