import os
import time
import random

from openai import (
    OpenAI,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)
from dotenv import load_dotenv

load_dotenv()


def get_client():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. Add it to your .env file."
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_retries=0,
        timeout=60.0
    )


def ask_model(model_id, messages, max_retries=3):
    client = get_client()

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0
            )

            return response

        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError
        ) as error:

            if attempt == max_retries:
                raise

            wait_time = min(2 ** attempt + random.random(), 30)

            print(
                f"Temporary API error: {type(error).__name__}. "
                f"Retrying in {wait_time:.1f} seconds..."
            )

            time.sleep(wait_time)