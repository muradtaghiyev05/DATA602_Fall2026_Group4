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

print("\nLoading TruthfulQA...")

truthfulqa = load_dataset(
    "truthfulqa/truthful_qa",
    "multiple_choice",
    split="validation"
)

print("Total questions:", len(truthfulqa))
print("Columns:", truthfulqa.column_names)

example = truthfulqa[0]
print("Question:", example["question"])
print("MC1 targets:", example["mc1_targets"])

four_option_indices = []

for idx, item in enumerate(truthfulqa):
    targets = item["mc1_targets"]

    if (
        len(targets["choices"]) == 4
        and targets["labels"].count(1) == 1
        and targets["labels"].count(0) == 3
    ):
        four_option_indices.append(idx)

print("Original four-option questions:", len(four_option_indices))


# Reproducibly sample 10 original four-option questions
truth_rng = random.Random(SEED + 1)
selected_truth_indices = truth_rng.sample(four_option_indices, 10)

truth_records = []

for idx in selected_truth_indices:
    item = truthfulqa[idx]
    targets = item["mc1_targets"]

    # Shuffle choices together with correctness labels
    # to prevent the correct answer from always appearing as A
    paired_choices = list(zip(targets["choices"], targets["labels"]))

    truth_rng.shuffle(paired_choices)

    choices, labels = zip(*paired_choices)
    correct_index = labels.index(1)
    
    truth_records.append({
        "question_id": f"truthfulqa_{idx}",
        "source": "TruthfulQA",
        "subject": "general",
        "question": item["question"],
        "options": json.dumps(dict(zip("ABCD", choices))),
        "correct_answer": "ABCD"[correct_index]
    })

truth_df = pd.DataFrame(truth_records)

assert len(truth_df) == 10
assert truth_df["question_id"].is_unique
assert truth_df["correct_answer"].isin(list("ABCD")).all()

truth_output = OUTPUT_DIR / "truthfulqa_10.csv"
truth_df.to_csv(truth_output, index=False)

print(f"Saved {len(truth_df)} TruthfulQA questions")


# Combine both question sources
pilot_df = pd.concat([df, truth_df], ignore_index=True)

assert len(pilot_df) == 50
assert pilot_df["question_id"].is_unique
assert (pilot_df["source"] == "MMLU").sum() == 40
assert (pilot_df["source"] == "TruthfulQA").sum() == 10

pilot_output = OUTPUT_DIR / "pilot_questions_50.csv"
pilot_df.to_csv(pilot_output, index=False)

print("\nFinal pilot dataset:")
print(pilot_df["source"].value_counts())
print(f"Total questions: {len(pilot_df)}")
print(f"Saved to {pilot_output}")
