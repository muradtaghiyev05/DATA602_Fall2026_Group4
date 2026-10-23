
import json
from pathlib import Path

import pandas as pd

# Input dataset
DATA_PATH = Path("data/sample/pilot_questions_50.csv")

REQUIRED_COLUMNS = [
    "question_id",
    "source",
    "subject",
    "question",
    "options",
    "correct_answer",
]

EXPECTED_SOURCES = {
    "MMLU": 40,
    "TruthfulQA": 10,
}

def validate_dataset():
    print("Loading pilot dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"\nTotal questions: {len(df)}")

    # 1. Check required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # 2. Check missing and blank values
    missing_counts = {}

    for col in REQUIRED_COLUMNS:
        missing_counts[col] = int(
            (df[col].isna() |
             df[col].astype(str).str.strip().eq("")).sum()
        )

    print("\nMissing values:")
    print(pd.Series(missing_counts))

    # 3. Check duplicates
    duplicate_ids = df["question_id"].duplicated().sum()

    normalized_questions = (
        df["question"].fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    duplicate_questions = normalized_questions.duplicated().sum()

    print(f"\nDuplicate question IDs: {duplicate_ids}")
    print(f"Duplicate question texts: {duplicate_questions}")

    # 4. Check source balance
    source_counts = df["source"].value_counts()

    print("\nQuestions by source:")
    print(source_counts)

    # 5. Check subject coverage
    print("\nUnique subjects by source:")
    print(df.groupby("source")["subject"].nunique())

    # 6. Validate answer options
    invalid_options = []
    invalid_answers = []

    for idx, row in df.iterrows():
        question_id = row["question_id"]

        try:
            options = json.loads(row["options"])

            if (
                not isinstance(options, dict)
                or set(options.keys()) != set("ABCD")
                or any(
                    not isinstance(value, str)
                    or not value.strip()
                    for value in options.values()
                )
                or len(set(options.values())) != 4
            ):
                invalid_options.append(question_id)

        except (TypeError, ValueError):
            invalid_options.append(question_id)

        if row["correct_answer"] not in list("ABCD"):
            invalid_answers.append(question_id)

    print(f"\nInvalid option sets: {len(invalid_options)}")
    print(f"Invalid correct answers: {len(invalid_answers)}")

    # 7. Overall quality checks
    checks = {
        "Exactly 50 questions": len(df) == 50,
        "No missing values": all(
            count == 0 for count in missing_counts.values()
        ),
        "Unique question IDs": duplicate_ids == 0,
        "Unique question texts": duplicate_questions == 0,
        "Valid answer options": len(invalid_options) == 0,
        "Valid answer letters": len(invalid_answers) == 0,
        "Correct source balance": all(
            source_counts.get(source, 0) == expected
            for source, expected in EXPECTED_SOURCES.items()
        ) and len(source_counts) == len(EXPECTED_SOURCES),
    }

    print("\nDATA QUALITY SUMMARY")
    print("-" * 40)

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    if all(checks.values()):
        print("\nAll dataset-quality checks passed!")
    else:
        print("\nSome checks failed. Review the results above.")


if __name__ == "__main__":
    validate_dataset()
