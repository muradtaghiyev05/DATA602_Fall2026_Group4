"""Make the blind audit file for checking the answer parser by hand.

Takes every v2 conversation the parser could not read (UNPARSEABLE or
REFUSED in either turn, now or live during collection) plus a random
10% of the rest, and writes
data/metadata/audit_sample.csv with empty label columns.

The parser's answers are left out on purpose, so the two human checkers
label blind and are not nudged toward agreeing with it.

Run from the repo root, after parse.py:
    python src/make_audit_sample.py
"""

import os

import pandas as pd

PARSED_FILE = "data/processed/pilot_parsed.csv"
OUTPUT = "data/metadata/audit_sample.csv"

SEED = 42
SAMPLE_FRACTION = 0.10


if __name__ == "__main__":

    # Don't overwrite a file the checkers may already have copied
    if os.path.exists(OUTPUT):
        raise SystemExit(f"{OUTPUT} already exists. Delete it first to make a new sample.")

    df = pd.read_csv(PARSED_FILE)
    df = df[df["pilot_version"] == "v2"]

    # Hard cases: not parsed now, or not parsed live during collection.
    # The second group is where the parser rules were extended after the
    # pilot, so they need a human check the most.
    not_parsed = (
        (df["parse_status_t1"] != "parsed")
        | (df["parse_status_t2"] != "parsed")
        | df["answer_t1_live"].isin(["UNPARSEABLE", "REFUSED"])
    )

    hard_cases = df[not_parsed]
    rest = df[~not_parsed]
    random_cases = rest.sample(frac=SAMPLE_FRACTION, random_state=SEED)

    audit = pd.concat([hard_cases, random_cases])

    # Shuffle so the hard cases are not all at the top of the file
    audit = audit.sample(frac=1, random_state=SEED)

    audit = audit[["conv_id", "response_t1", "response_t2"]].copy()
    audit["label_t1"] = ""
    audit["label_t2"] = ""

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    audit.to_csv(OUTPUT, index=False)

    print(f"Hard cases (UNPARSEABLE/REFUSED): {len(hard_cases)}")
    print(f"Random {SAMPLE_FRACTION:.0%} of the rest: {len(random_cases)} of {len(rest)}")
    print(f"Saved {len(audit)} conversations to {OUTPUT}")
    print()
    print("Each checker makes their own copy, for example:")
    print("  data/metadata/audit_labels_1.csv and data/metadata/audit_labels_2.csv")
    print("and fills label_t1 and label_t2 with the letter the model chose")
    print("(A, B, C or D), or REFUSED, or UNPARSEABLE. Label alone, without")
    print("talking to the other checker.")
