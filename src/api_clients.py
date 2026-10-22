import os

from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

def get_client():

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:

        raise ValueError(

            "OPENROUTER_API_KEY is missing. Add it to your .env file."

        )

    client = OpenAI(

        base_url="https://openrouter.ai/api/v1",

        api_key=api_key

    )

    return client

def ask_model(model_id, messages):

    client = get_client()

    response = client.chat.completions.create(

        model=model_id,

        messages=messages,

        temperature=0

    )

    return response