import csv
import ast
import random
import os
import json

from datetime import datetime, timezone

from api_clients import ask_model
from prompts import PUSHBACK_TYPES, get_pushback

QUESTIONS_FILE = "data/sample/mmlu_test_5.csv"

CSV_OUTPUT = "data/raw/pilot_results.csv"
JSONL_OUTPUT = "data/raw/pilot_raw.jsonl"

MODELS = [

    "openai/gpt-4o-mini",

    "google/gemini-2.5-flash",

    "x-ai/grok-4.3"

]

# True = preview only, no API requests
# False = send real API requests
DRY_RUN = False 

def load_questions(file_path):
    questions = []

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            row["options"] = ast.literal_eval(row["options"])

            if len(row["options"]) != 4:
                raise ValueError(
                    f"Question {row['question_id']} does not have 4 options"
                )

            questions.append(row)

    return questions


def format_question(row):
    letters = ["A", "B", "C", "D"]

    question_text = row["question"] + "\n\n"

    for letter, option in zip(letters, row["options"]):
        question_text += f"{letter}. {option}\n"

    question_text += "\nRespond with exactly: Final answer: <letter>"

    return question_text


def choose_wrong_answer(correct_answer, question_id):
    letters = ["A", "B", "C", "D"]

    wrong_options = [
        letter for letter in letters
        if letter != correct_answer
    ]

    rng = random.Random(f"42:{question_id}")

    return rng.choice(wrong_options)


def is_already_collected(question_id, model_id, pushback_type):

    if not os.path.exists(JSONL_OUTPUT):
        return False

    with open(JSONL_OUTPUT, "r", encoding="utf-8") as file:

        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if (
                record["question_id"] == question_id
                and record["model_id"] == model_id
                and record["pushback_type"] == pushback_type
            ):
                return True

    return False


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



def rebuild_csv_from_jsonl():

    if not os.path.exists(JSONL_OUTPUT):
        return

    records = {}

    with open(JSONL_OUTPUT, "r", encoding="utf-8") as file:

        for line in file:
            if not line.strip():
                continue

            raw = json.loads(line)

            key = (
                raw["question_id"],
                raw["model_id"],
                raw["pushback_type"]
            )

            records[key] = {
                "question_id": raw["question_id"],
                "source": raw["source"],
                "subject": raw["subject"],
                "model_id": raw["model_id"],
                "pushback_type": raw["pushback_type"],
                "correct_answer": raw["correct_answer"],
                "suggested_wrong_answer": raw["suggested_wrong_answer"],
                "question_text": raw["messages"][0]["content"],
                "pushback_text": raw["messages"][2]["content"],
                "response_t1": raw["response_t1_raw"]["choices"][0]["message"]["content"],
                "response_t2": raw["response_t2_raw"]["choices"][0]["message"]["content"],
                "timestamp": raw["timestamp"]
            }

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


# ==========================================
# RUN ONE EXPERIMENT
# ==========================================

def run_experiment(question, pushback_type, model_id):

    question_id = question["question_id"]
    correct_answer = question["correct_answer"]

    if not DRY_RUN and is_already_collected(
        question_id,
        model_id,
        pushback_type
    ):
        print(
            f"SKIPPED: {question_id} / "
            f"{model_id} / {pushback_type}"
        )
        return

    wrong_answer = choose_wrong_answer(
        correct_answer,
        question_id
    )

    question_text = format_question(question)

    pushback = get_pushback(
        pushback_type,
        wrong_answer
    )

    print("\n------------------------------")
    print("Question:", question_id)
    print("Model:", model_id)
    print("Condition:", pushback_type)
    print("Correct answer:", correct_answer)
    print("Suggested wrong answer:", wrong_answer)
    print("Pushback:", pushback)

    if DRY_RUN:
        print("DRY RUN — No API request sent.")
        return


    messages = [
        {
            "role": "user",
            "content": question_text
        }
    ]

    response1 = ask_model(
        model_id=model_id,
        messages=messages
    )

    answer1 = response1.choices[0].message.content


    messages.append({
        "role": "assistant",
        "content": answer1
    })

    messages.append({
        "role": "user",
        "content": pushback
    })

    response2 = ask_model(
        model_id=model_id,
        messages=messages
    )

    answer2 = response2.choices[0].message.content

    print("Initial response:", answer1)
    print("Final response:", answer2)

    timestamp = datetime.now(timezone.utc).isoformat()

    raw_record = {
        "question_id": question_id,
        "source": question["source"],
        "subject": question["subject"],
        "correct_answer": correct_answer,
        "model_id": model_id,
        "pushback_type": pushback_type,
        "suggested_wrong_answer": wrong_answer,
        "timestamp": timestamp,
        "messages": messages,
        "response_t1_raw": response1.model_dump(mode="json"),
        "response_t2_raw": response2.model_dump(mode="json")
    }

    save_raw_response(raw_record)


    result = {
        "question_id": question_id,
        "source": question["source"],
        "subject": question["subject"],
        "model_id": model_id,
        "pushback_type": pushback_type,
        "correct_answer": correct_answer,
        "suggested_wrong_answer": wrong_answer,
        "question_text": question_text,
        "pushback_text": pushback,
        "response_t1": answer1,
        "response_t2": answer2,
        "timestamp": timestamp
    }

    save_result(result)

    print("Result saved successfully!")


if __name__ == "__main__":

    if not DRY_RUN:
        rebuild_csv_from_jsonl()

    questions = load_questions(QUESTIONS_FILE)

    print("Total questions loaded:", len(questions))
    print("Models configured:", len(MODELS))
    print("DRY_RUN:", DRY_RUN)

    for question in questions:
        for model_id in MODELS:

            print(f"\n===== MODEL: {model_id} =====")

            for pushback_type in PUSHBACK_TYPES:
                run_experiment(
                    question,
                    pushback_type,
                    model_id
                )

    print("\nAll model-condition combinations completed!")