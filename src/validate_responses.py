"""Data-quality checks on the parsed pilot responses.

Maheer's src/validate.py checks the question file. This script checks the
model responses in data/processed/pilot_parsed.csv and prints a summary
table for the progress report.

Run from the repo root, after parse.py:
    python src/validate_responses.py
"""

import json
import os

import pandas as pd

PARSED_FILE = "data/processed/pilot_parsed.csv"
QUESTIONS_FILE = "data/sample/pilot_questions_50.csv"
ERRORS_FILE = "data/raw/errors_v2.jsonl"

MODELS = ["openai/gpt-4o-mini", "google/gemini-2.5-flash", "x-ai/grok-4.3"]
PUSHBACK_TYPES = ["control", "doubt", "confident", "authority", "emotional"]
ALLOWED_ANSWERS = ["A", "B", "C", "D", "REFUSED", "UNPARSEABLE"]
ALLOWED_DIRECTIONS = ["toward_wrong", "toward_correct", "unknown_t1"]
LETTERS = ["A", "B", "C", "D"]

summary = []


def record(check, version, problems, total, note=""):
    # A check passes when it finds 0 problems
    summary.append({
        "check": check,
        "version": version,
        "problems": problems,
        "out_of": total,
        "status": "PASS" if problems == 0 else "FLAG",
        "note": note,
    })


def check_duplicates(df, version):
    duplicated = df[df["conv_id"].duplicated(keep=False)]
    record("Duplicate conv_id", version, duplicated["conv_id"].nunique(), len(df))

    if len(duplicated):
        print(f"\n[{version}] Duplicate conv_ids:")
        print(duplicated["conv_id"].value_counts().to_string())


def check_coverage(df, version, question_ids):
    # Every question should have every model x pushback type
    expected = {
        (q, m, p)
        for q in question_ids
        for m in MODELS
        for p in PUSHBACK_TYPES
    }
    found = set(zip(df["question_id"], df["model_id"], df["pushback_type"]))
    missing = sorted(expected - found)

    record(
        "Coverage (question x model x pushback)", version,
        len(missing), len(expected),
        f"{len(question_ids)} questions"
    )

    if missing:
        print(f"\n[{version}] Missing conversations:")
        for question_id, model_id, pushback_type in missing:
            print(f"  {question_id} / {model_id} / {pushback_type}")


def check_allowed_values(df, version):
    checks = [
        ("Allowed values: answer_t1", "answer_t1", ALLOWED_ANSWERS),
        ("Allowed values: answer_t2", "answer_t2", ALLOWED_ANSWERS),
        ("Allowed values: pushback_type", "pushback_type", PUSHBACK_TYPES),
        ("Allowed values: model_id", "model_id", MODELS),
        ("Allowed values: pushed_direction", "pushed_direction", ALLOWED_DIRECTIONS),
        ("Allowed values: correct_answer", "correct_answer", LETTERS),
        ("Allowed values: pushed_letter", "pushed_letter", LETTERS),
    ]

    for check, column, allowed in checks:
        bad = df[~df[column].isin(allowed)]
        note = ""
        if len(bad):
            note = "found: " + ", ".join(sorted(bad[column].astype(str).unique()))
        record(check, version, len(bad), len(df), note)


def check_push_logic(df, version):
    # toward_wrong: the pushed letter must be wrong and differ from turn 1
    toward_wrong = df[df["pushed_direction"] == "toward_wrong"]
    same_as_t1 = toward_wrong[toward_wrong["pushed_letter"] == toward_wrong["answer_t1"]]
    record(
        "toward_wrong: pushed_letter != answer_t1", version,
        len(same_as_t1), len(toward_wrong),
        "user 'pushed' the model's own wrong answer" if len(same_as_t1) else ""
    )

    pushed_correct = toward_wrong[toward_wrong["pushed_letter"] == toward_wrong["correct_answer"]]
    record(
        "toward_wrong: pushed_letter is wrong", version,
        len(pushed_correct), len(toward_wrong)
    )

    # toward_correct: the pushed letter must be the correct one
    toward_correct = df[df["pushed_direction"] == "toward_correct"]
    not_correct = toward_correct[toward_correct["pushed_letter"] != toward_correct["correct_answer"]]
    record(
        "toward_correct: pushed_letter is correct", version,
        len(not_correct), len(toward_correct)
    )

    # Design rule: a wrong turn 1 should be pushed toward the correct letter
    wrong_t1 = df[df["parse_status_t1"].eq("parsed") & (df["answer_t1"] != df["correct_answer"])]
    not_corrected = wrong_t1[wrong_t1["pushed_direction"] != "toward_correct"]
    record(
        "Design rule: wrong turn 1 pushed toward correct", version,
        len(not_corrected), len(wrong_t1)
    )


def check_live_parse(df, version):
    # collect.py parsed turn 1 live to choose the push. Did parse.py agree?
    if df["answer_t1_live"].isna().all():
        return

    differ = df[df["answer_t1_live"] != df["answer_t1"]]
    note = ""
    if len(differ):
        changes = (differ["answer_t1_live"] + " -> " + differ["answer_t1"]).value_counts()
        note = "; ".join(f"{k} ({v})" for k, v in changes.items())
    record("answer_t1_live matches answer_t1", version, len(differ), len(df), note)


def check_responses(df, version):
    for turn in ["t1", "t2"]:
        text = df[f"response_{turn}"]
        empty = text.isna() | (text.astype(str).str.strip() == "")
        record(f"Empty response {turn}", version, int(empty.sum()), len(df))

        not_stop = df[df[f"finish_reason_{turn}"] != "stop"]
        note = ""
        if len(not_stop):
            note = "found: " + ", ".join(not_stop[f"finish_reason_{turn}"].astype(str).unique())
        record(f"finish_reason {turn} != stop", version, len(not_stop), len(df), note)

        unparsed = df[df[f"parse_status_{turn}"] != "parsed"]
        record(f"Unparsed answer {turn}", version, len(unparsed), len(df))


def check_model_returned(df, version):
    differ = df[df["model_returned"] != df["model_id"]]
    note = ""
    if len(differ):
        note = "found: " + ", ".join(differ["model_returned"].astype(str).unique())
    record("model_returned matches model_id", version, len(differ), len(df), note)


def check_missing_values(df, version):
    # Columns that should never be empty for this version
    required = [
        "conv_id", "question_id", "model_id", "pushback_type", "correct_answer",
        "pushed_letter", "pushed_direction", "answer_t1", "answer_t2",
        "prompt_tokens", "completion_tokens", "cost_usd",
        "finish_reason_t1", "finish_reason_t2", "model_returned",
    ]
    if version == "v2":
        required += ["answer_t1_live", "latency_ms_t1", "latency_ms_t2"]

    missing = df[required].isna().sum()
    missing = missing[missing > 0]
    note = ", ".join(f"{col} ({n})" for col, n in missing.items())
    record("Missing values in required columns", version, int(missing.sum()), len(df) * len(required), note)


def check_errors_file(df_v2):
    if not os.path.exists(ERRORS_FILE):
        record("Rows in errors_v2.jsonl", "v2", 0, len(df_v2), "no errors file")
        return

    errors = []
    with open(ERRORS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                errors.append(json.loads(line))

    collected = set(df_v2["conv_id"])
    not_recovered = [e for e in errors if e["conv_id"] not in collected]

    record(
        "Rows in errors_v2.jsonl", "v2", len(errors), len(df_v2),
        f"{len(errors) - len(not_recovered)} recovered by a rerun"
    )
    record("Errors never recovered", "v2", len(not_recovered), len(errors))

    if errors:
        print("\nLogged collection errors:")
        for e in errors:
            status = "recovered" if e["conv_id"] in collected else "MISSING"
            print(f"  {e['conv_id']}: {e['error_type']}: {e['error_message']} [{status}]")


def print_answer_balance(questions):
    # Correct-answer letter per source. A balanced 4-option test is ~25% each.
    table = pd.crosstab(questions["source"], questions["correct_answer"], margins=True, margins_name="Total")
    table = table.reindex(columns=LETTERS + ["Total"], fill_value=0)

    print("\nCorrect-answer letter balance by source (question file):")
    print(table.to_string())

    for source, group in questions.groupby("source"):
        top_share = group["correct_answer"].value_counts(normalize=True).max()
        top_letter = group["correct_answer"].value_counts().idxmax()
        record(
            f"Answer balance: {source}", "questions",
            int(top_share > 0.5), 1,
            f"most common letter {top_letter} = {top_share:.0%}"
        )


if __name__ == "__main__":

    df = pd.read_csv(PARSED_FILE)
    questions = pd.read_csv(QUESTIONS_FILE)

    print(f"Loaded {len(df)} parsed conversations from {PARSED_FILE}")
    print(df["pilot_version"].value_counts().to_string())

    for version, group in df.groupby("pilot_version"):

        # v2 should cover the 50 pilot questions. v1 used the old test file,
        # so check it against its own questions.
        if version == "v2":
            question_ids = questions["question_id"].tolist()
        else:
            question_ids = group["question_id"].unique().tolist()

        check_duplicates(group, version)
        check_coverage(group, version, question_ids)
        check_allowed_values(group, version)
        check_push_logic(group, version)
        check_live_parse(group, version)
        check_responses(group, version)
        check_model_returned(group, version)
        check_missing_values(group, version)

    check_errors_file(df[df["pilot_version"] == "v2"])
    print_answer_balance(questions)

    print("\n===== VALIDATION SUMMARY =====")
    summary_df = pd.DataFrame(summary)
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(summary_df.to_string(index=False))

    flags = summary_df[summary_df["status"] == "FLAG"]
    print(f"\n{len(summary_df)} checks, {len(flags)} flagged.")
