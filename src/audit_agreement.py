"""Compare two human-labeled audit files with the parser.

Reports:
- parser accuracy against each checker, and against the cases where
  both checkers agree
- Cohen's kappa between the two checkers (agreement beyond chance)
- every disagreement, so we can look at it

Run from the repo root, after both checkers filled their copies:
    python src/audit_agreement.py
    python src/audit_agreement.py path/to/labels_1.csv path/to/labels_2.csv
"""

import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

PARSED_FILE = "data/processed/pilot_parsed.csv"
LABELS_1 = "data/metadata/audit_labels_1.csv"
LABELS_2 = "data/metadata/audit_labels_2.csv"

ALLOWED_LABELS = ["A", "B", "C", "D", "REFUSED", "UNPARSEABLE"]


def load_labels(path):
    labels = pd.read_csv(path, dtype=str, keep_default_na=False)

    # One row per (conversation, turn), so both turns are scored together
    long = pd.concat([
        labels[["conv_id", "label_t1"]].rename(columns={"label_t1": "label"}).assign(turn="t1"),
        labels[["conv_id", "label_t2"]].rename(columns={"label_t2": "label"}).assign(turn="t2"),
    ])
    long["label"] = long["label"].str.strip().str.upper()

    blank = (long["label"] == "").sum()
    bad = long[(long["label"] != "") & ~long["label"].isin(ALLOWED_LABELS)]

    if blank:
        print(f"WARNING: {path} has {blank} empty labels. They are left out.")
    if len(bad):
        print(f"WARNING: {path} has labels that are not allowed, left out:")
        print(bad.to_string(index=False))

    return long[long["label"].isin(ALLOWED_LABELS)]


def load_parser_answers():
    parsed = pd.read_csv(PARSED_FILE)
    parsed = parsed[parsed["pilot_version"] == "v2"]

    return pd.concat([
        parsed[["conv_id", "answer_t1"]].rename(columns={"answer_t1": "parser"}).assign(turn="t1"),
        parsed[["conv_id", "answer_t2"]].rename(columns={"answer_t2": "parser"}).assign(turn="t2"),
    ])


if __name__ == "__main__":

    path_1 = sys.argv[1] if len(sys.argv) > 2 else LABELS_1
    path_2 = sys.argv[2] if len(sys.argv) > 2 else LABELS_2

    checker_1 = load_labels(path_1).rename(columns={"label": "checker_1"})
    checker_2 = load_labels(path_2).rename(columns={"label": "checker_2"})
    parser = load_parser_answers()

    # Only items both checkers labeled
    merged = checker_1.merge(checker_2, on=["conv_id", "turn"])
    merged = merged.merge(parser, on=["conv_id", "turn"], how="left")

    missing_parser = merged["parser"].isna().sum()
    if missing_parser:
        print(f"WARNING: {missing_parser} labeled items have no parser answer (conv_id not in v2).")
        merged = merged[merged["parser"].notna()]

    n = len(merged)
    print(f"Labeled by both checkers: {n} replies "
          f"from {merged['conv_id'].nunique()} conversations")

    # Agreement between the two humans
    agree = merged["checker_1"] == merged["checker_2"]
    kappa = cohen_kappa_score(merged["checker_1"], merged["checker_2"])

    print("\n===== CHECKER AGREEMENT =====")
    print(f"Raw agreement: {agree.mean():.1%} ({agree.sum()}/{n})")
    print(f"Cohen's kappa: {kappa:.3f}")

    # Parser against the humans
    acc_1 = (merged["parser"] == merged["checker_1"]).mean()
    acc_2 = (merged["parser"] == merged["checker_2"]).mean()

    consensus = merged[agree]
    acc_consensus = (consensus["parser"] == consensus["checker_1"]).mean()

    print("\n===== PARSER ACCURACY =====")
    print(f"vs checker 1:  {acc_1:.1%}")
    print(f"vs checker 2:  {acc_2:.1%}")
    print(f"vs consensus:  {acc_consensus:.1%} "
          f"(on {len(consensus)} replies where both checkers agree)")

    print("\nParser vs consensus, by turn:")
    for turn, group in consensus.groupby("turn"):
        turn_acc = (group["parser"] == group["checker_1"]).mean()
        print(f"  {turn}: {turn_acc:.1%} (n={len(group)})")

    # Show every disagreement so we can decide who was right
    disagreements = merged[~agree | (merged["parser"] != merged["checker_1"])]
    if len(disagreements):
        print(f"\n===== DISAGREEMENTS ({len(disagreements)}) =====")
        print(disagreements[["conv_id", "turn", "checker_1", "checker_2", "parser"]]
              .sort_values(["conv_id", "turn"]).to_string(index=False))
