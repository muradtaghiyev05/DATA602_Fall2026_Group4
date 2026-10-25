# DATA602 Fall 2026, Group 4: LLM Sycophancy Dataset

A documented dataset that measures **LLM sycophancy**: when a user pushes back on an answer, does the model drop a correct one?

Each conversation has two turns:

- **Turn 1:** the model answers a 4-option multiple-choice question (MMLU or TruthfulQA).
- **Turn 2:** the user sends one of 5 follow-ups (`control`, `doubt`, `confident`, `authority`, `emotional`).

If turn 1 was correct, the pushback points to a wrong letter. If it was wrong, it points to the correct letter. That way, dropping a right answer (sycophancy) can be told apart from accepting a real correction.

- **Pilot (v2):** 50 questions × 3 models × 5 pushback types = 750 conversations.
- **Models:** `openai/gpt-4o-mini`, `google/gemini-2.5-flash`, `x-ai/grok-4.3`, all through OpenRouter at temperature 0.
- **Final target:** 600 questions, 9,000 conversations.

Team: Maheer (question sampling), Narmin (collection), Murad (parsing, validation, EDA, documentation).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then put your OpenRouter key in `.env`:

```
OPENROUTER_API_KEY=your-key-here
```

`.env` is git-ignored. Never commit it.

## Run order

Run every step from the repo root.

| Step | Command | Output |
|---|---|---|
| 1. Sample questions | `python src/prepare_questions.py` | `data/sample/pilot_questions_50.csv` |
| 2. Collect conversations | `python src/collect.py` | `data/raw/pilot_raw_v2.jsonl`, `data/raw/pilot_results_v2.csv`, `data/raw/errors_v2.jsonl` |
| 3. Parse | `python src/parse.py` | `data/processed/pilot_parsed.csv` and a parse-rate table |
| 4. Validate responses | `python src/validate_responses.py` | Data-quality summary table (printed) |
| 5. EDA notebook | `jupyter notebook notebooks/pilot_eda.ipynb`, then Run All | Figures in `reports/figures/` |

### Collection settings (`src/collect.py`)

- **`DRY_RUN = True`** prints every conversation without calling the API. It's free.
- **`LIMIT = 2`** runs only the first 2 questions (30 conversations), for a quick test. `LIMIT = None` runs all of them.
- **Resume:** the script skips conversations already in `pilot_raw_v2.jsonl`. If a run stops, just run it again.
- **Errors:** failed conversations are logged to `errors_v2.jsonl` and retried on the next run.

### Parser audit (optional)

This checks the parser against human labels.

1. **Make the sample:** `python src/make_audit_sample.py` writes `data/metadata/audit_sample.csv`. It contains replies the parser couldn't read plus a random 10% of the rest, without the parser's answers.
2. **Label it:** two people each save a copy as `data/metadata/audit_labels_1.csv` and `audit_labels_2.csv`. Each fills `label_t1` and `label_t2` independently with the letter the model chose (`A`–`D`, `REFUSED` or `UNPARSEABLE`).
3. **Score it:** `python src/audit_agreement.py` reports parser accuracy and Cohen's kappa between the two labelers.

## Repo layout

```
src/
  prepare_questions.py   question sampling (MMLU + TruthfulQA, seed 42)
  prompts.py             pushback messages, format reminder, prompt version
  api_clients.py         OpenRouter client with retry and backoff
  answer_parser.py       extract_answer(), shared by collect.py and parse.py
  collect.py             runs the two-turn conversations
  parse.py               raw logs -> one clean row per conversation
  validate_responses.py  data-quality checks on the responses
  make_audit_sample.py   blind audit sample for human labeling
  audit_agreement.py     parser accuracy and labeler agreement
data/
  sample/                question files
  raw/                   raw API logs (v1 = pilot_raw.jsonl, v2 = pilot_raw_v2.jsonl)
  processed/             pilot_parsed.csv
  metadata/              audit sample and labels
notebooks/pilot_eda.ipynb
reports/figures/         figures used in the report
Data_Dictionary.md       every field in pilot_parsed.csv
```

See [`Data_Dictionary.md`](Data_Dictionary.md) for field definitions, missing-value rules and known data issues.

## AI Usage Disclosure

We used **Cursor** (an AI coding assistant) to help write parts of this project, including:

- `src/answer_parser.py`, `src/parse.py`, `src/validate_responses.py`, `src/make_audit_sample.py` and `src/audit_agreement.py`
- the pipeline fixes in `src/collect.py` and `src/prompts.py`
- `notebooks/pilot_eda.ipynb`
- this README and `Data_Dictionary.md`

We reviewed all AI-assisted code and text, and tested it:

- the parser on real pilot replies;
- a dry run and a 30-conversation test run before the full collection;
- the notebook run top to bottom after a kernel restart.

The design, the choice of models and questions, and the interpretation of results are our own.

The parser audit labels in `data/metadata/audit_labels_1.csv` and `audit_labels_2.csv` were made by: _\<labeler 1\>_ and _\<labeler 2\>_.
