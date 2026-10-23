import csv
import random
import os
import json
import time

from datetime import datetime, timezone

from api_clients import ask_model
from prompts import PUSHBACK_TYPES, PROMPT_VERSION, get_pushback
from answer_parser import extract_answer

QUESTIONS_FILE = "data/sample/pilot_questions_50.csv"

CSV_OUTPUT = "data/raw/pilot_results_v2.csv"
JSONL_OUTPUT = "data/raw/pilot_raw_v2.jsonl"
ERRORS_OUTPUT = "data/raw/errors_v2.jsonl"

SCHEMA_VERSION = "v2"

LETTERS = ["A", "B", "C", "D"]

MODELS = [

    "openai/gpt-4o-mini",

    "google/gemini-2.5-flash",

    "x-ai/grok-4.3"

]

# True = preview only, no API requests
# False = send real API requests
DRY_RUN = False 

# None = all questions
# A number (e.g. 2) = only the first N questions, for a quick test
LIMIT = None

def load_questions(file_path):
    questions = []

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            row["options"] = json.loads(row["options"])

            if sorted(row["options"].keys()) != LETTERS:
                raise ValueError(
                    f"Question {row['question_id']} does not have options A-D"
                )

            questions.append(row)

    return questions


def format_question(row):

    question_text = row["question"] + "\n\n"

    for letter in LETTERS:
        question_text += f"{letter}. {row['options'][letter]}\n"

    question_text += "\nRespond with exactly: Final answer: <letter>"

    return question_text


def choose_wrong_answer(correct_answer, question_id):

    wrong_options = [
        letter for letter in LETTERS
        if letter != correct_answer
    ]

    rng = random.Random(f"42:{question_id}")

    return rng.choice(wrong_options)


def choose_pushed_letter(answer_t1, correct_answer, question_id):

    # Design rule: if turn 1 is correct, push toward a wrong letter.
    # If turn 1 is wrong, push toward the correct letter.
    if answer_t1 == correct_answer:
        return choose_wrong_answer(correct_answer, question_id), "toward_wrong"

    if answer_t1 in LETTERS:
        return correct_answer, "toward_correct"

    # REFUSED or UNPARSEABLE: still run turn 2, but flag it
    return choose_wrong_answer(correct_answer, question_id), "unknown_t1"


def make_conv_id(question_id, model_id, pushback_type):

    model_short_name = model_id.split("/")[-1]

    return f"{question_id}__{model_short_name}__{pushback_type}"


def load_collected_ids():

    collected_ids = set()

    if not os.path.exists(JSONL_OUTPUT):
        return collected_ids

    with open(JSONL_OUTPUT, "r", encoding="utf-8") as file:

        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            collected_ids.add(record["conv_id"])

    return collected_ids


def sum_usage(raw1, raw2, field):

    # Returns None if either turn is missing the field
    values = [
        (raw.get("usage") or {}).get(field)
        for raw in [raw1, raw2]
    ]

    if None in values:
        return None

    return sum(values)


def make_csv_row(raw):

    # Everything except the nested message list and raw API responses
    row = {
        key: value for key, value in raw.items()
        if key not in ["messages", "response_t1_raw", "response_t2_raw"]
    }

    row["question_text"] = raw["messages"][0]["content"]
    row["pushback_text"] = raw["messages"][2]["content"]
    row["response_t1"] = raw["response_t1_raw"]["choices"][0]["message"]["content"]
    row["response_t2"] = raw["response_t2_raw"]["choices"][0]["message"]["content"]

    return row


def save_result(result):

    os.makedirs("data/raw", exist_ok=True)

    file_exists = os.path.exists(CSV_OUTPUT)

    with open(CSV_OUTPUT, "a", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=result.keys()
        )

        if not file_exists or os.path.getsize(CSV_OUTPUT) == 0:
            writer.writeheader()

        writer.writerow(result)


# ==========================================
# SAVE RAW JSONL
# ==========================================

def save_raw_response(record):

    os.makedirs("data/raw", exist_ok=True)

    with open(JSONL_OUTPUT, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def save_error(conv_id, error):

    os.makedirs("data/raw", exist_ok=True)

    record = {
        "conv_id": conv_id,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    with open(ERRORS_OUTPUT, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def rebuild_csv_from_jsonl():

    if not os.path.exists(JSONL_OUTPUT):
        return

    records = {}

    with open(JSONL_OUTPUT, "r", encoding="utf-8") as file:

        for line in file:
            if not line.strip():
                continue

            raw = json.loads(line)

            records[raw["conv_id"]] = make_csv_row(raw)

    if not records:
        return

    os.makedirs("data/raw", exist_ok=True)

    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(next(iter(records.values())).keys())
        )

        writer.writeheader()
        writer.writerows(records.values())

    print(f"CSV synchronized: {len(records)} conversations")


def print_parse_rates():

    if not os.path.exists(JSONL_OUTPUT):
        return

    counts = {}

    with open(JSONL_OUTPUT, "r", encoding="utf-8") as file:

        for line in file:
            if not line.strip():
                continue

            raw = json.loads(line)

            model_counts = counts.setdefault(
                raw["model_id"],
                {"total": 0, "t1": 0, "t2": 0}
            )

            model_counts["total"] += 1

            for turn in ["t1", "t2"]:
                text = raw[f"response_{turn}_raw"]["choices"][0]["message"]["content"]

                if extract_answer(text) in LETTERS:
                    model_counts[turn] += 1

    print("\nParse rate per model:")

    for model_id, model_counts in counts.items():
        total = model_counts["total"]
        print(
            f"  {model_id}: "
            f"turn 1 {model_counts['t1']}/{total}, "
            f"turn 2 {model_counts['t2']}/{total}"
        )


# ==========================================
# RUN ONE EXPERIMENT
# ==========================================

def run_experiment(question, pushback_type, model_id):

    question_id = question["question_id"]
    correct_answer = question["correct_answer"]

    conv_id = make_conv_id(question_id, model_id, pushback_type)

    question_text = format_question(question)

    print("\n------------------------------")
    print("Conversation:", conv_id)
    print("Correct answer:", correct_answer)

    if DRY_RUN:
        wrong_answer = choose_wrong_answer(correct_answer, question_id)

        print("Pushed letter is chosen after turn 1:")
        print("  if turn 1 is correct ->", wrong_answer)
        print("  if turn 1 is wrong   ->", correct_answer)
        print("Pushback (example):", get_pushback(pushback_type, wrong_answer))
        print("DRY RUN — No API request sent.")
        return


    messages = [
        {
            "role": "user",
            "content": question_text
        }
    ]

    start = time.perf_counter()

    response1 = ask_model(
        model_id=model_id,
        messages=messages
    )

    latency_ms_t1 = round((time.perf_counter() - start) * 1000)

    answer1 = response1.choices[0].message.content or ""

    answer_t1_live = extract_answer(answer1)

    pushed_letter, pushed_direction = choose_pushed_letter(
        answer_t1_live,
        correct_answer,
        question_id
    )

    pushback = get_pushback(
        pushback_type,
        pushed_letter
    )

    print("Turn 1 answer:", answer_t1_live)
    print("Pushed letter:", pushed_letter, f"({pushed_direction})")
    print("Pushback:", pushback)


    messages.append({
        "role": "assistant",
        "content": answer1
    })

    messages.append({
        "role": "user",
        "content": pushback
    })

    start = time.perf_counter()

    response2 = ask_model(
        model_id=model_id,
        messages=messages
    )

    latency_ms_t2 = round((time.perf_counter() - start) * 1000)

    answer2 = response2.choices[0].message.content or ""

    print("Initial response:", answer1)
    print("Final response:", answer2)

    timestamp = datetime.now(timezone.utc).isoformat()

    raw1 = response1.model_dump(mode="json")
    raw2 = response2.model_dump(mode="json")

    raw_record = {
        "conv_id": conv_id,
        "question_id": question_id,
        "source": question["source"],
        "subject": question["subject"],
        "correct_answer": correct_answer,
        "model_id": model_id,
        "pushback_type": pushback_type,
        "pushed_letter": pushed_letter,
        "pushed_direction": pushed_direction,
        "answer_t1_live": answer_t1_live,
        "latency_ms_t1": latency_ms_t1,
        "latency_ms_t2": latency_ms_t2,
        "prompt_tokens": sum_usage(raw1, raw2, "prompt_tokens"),
        "completion_tokens": sum_usage(raw1, raw2, "completion_tokens"),
        "cost_usd": sum_usage(raw1, raw2, "cost"),
        "finish_reason_t1": raw1["choices"][0]["finish_reason"],
        "finish_reason_t2": raw2["choices"][0]["finish_reason"],
        "model_returned": raw1.get("model"),
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "timestamp": timestamp,
        "messages": messages,
        "response_t1_raw": raw1,
        "response_t2_raw": raw2
    }

    save_raw_response(raw_record)

    save_result(make_csv_row(raw_record))

    print("Result saved successfully!")


if __name__ == "__main__":

    if not DRY_RUN:
        rebuild_csv_from_jsonl()

    questions = load_questions(QUESTIONS_FILE)

    if LIMIT:
        questions = questions[:LIMIT]

    # Read the JSONL once, so resuming does not re-read it every time
    collected_ids = load_collected_ids()

    print("Total questions loaded:", len(questions))
    print("Models configured:", len(MODELS))
    print("Already collected:", len(collected_ids))
    print("DRY_RUN:", DRY_RUN)

    for question in questions:
        for model_id in MODELS:

            print(f"\n===== MODEL: {model_id} =====")

            for pushback_type in PUSHBACK_TYPES:

                conv_id = make_conv_id(
                    question["question_id"],
                    model_id,
                    pushback_type
                )

                if not DRY_RUN and conv_id in collected_ids:
                    print(f"SKIPPED: {conv_id}")
                    continue

                # One failed conversation should not stop the whole run
                try:
                    run_experiment(
                        question,
                        pushback_type,
                        model_id
                    )

                    if not DRY_RUN:
                        collected_ids.add(conv_id)

                except Exception as error:
                    print(f"ERROR: {conv_id}: {type(error).__name__}: {error}")
                    save_error(conv_id, error)

    print("\nAll model-condition combinations completed!")

    if not DRY_RUN:
        print_parse_rates()
