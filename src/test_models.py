from api_clients import ask_model

MODELS = [
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
    "x-ai/grok-4.3"
]

messages = [
    {
        "role": "user",
        "content": (
            "What is the capital of France?\n"
            "A. London\n"
            "B. Berlin\n"
            "C. Paris\n"
            "D. Rome\n\n"
            "Respond with exactly: Final answer: <letter>"
        )
    }
]

for model_id in MODELS:
    print(f"\nTesting: {model_id}")

    try:
        response = ask_model(
            model_id=model_id,
            messages=messages
        )

        print("SUCCESS!")
        print("Response:", response.choices[0].message.content)

    except Exception as error:
        print("ERROR:", type(error).__name__)
        print("Details:", str(error)[:300])

print("\nModel testing completed!")