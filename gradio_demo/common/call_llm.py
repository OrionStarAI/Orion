import json
import os
from typing import Dict, Generator, List

import requests

# Environment variables for backend URL and model name
BACKEND_HOST = os.getenv("BACKEND_HOST")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("API_KEY")


# Custom headers for the API request
HEADERS = {
    "orionstar-api-key": API_KEY,
}


def chat_stream_generator(
    messages: List[Dict[str, str]], endpoint: str, temperature: float = 0.7
) -> Generator[str, None, None]:
    """Generator function to stream chat responses from the backend."""

    payload = {
        "model": MODEL_NAME,
        "stream": True,
        "messages": messages,
        "temperature": temperature,
    }

    try:
        with requests.post(
            BACKEND_HOST + endpoint,
            json=payload,
            headers=HEADERS,
            timeout=60 * 10,
            stream=True,
        ) as response:
            if response.status_code != 200:
                yield f"[model server error]. response text: {response.text}"
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                if isinstance(line, bytes):
                    line = line.decode("utf-8")

                line = line.replace("data: ", "")
                if line == "[DONE]":
                    return

                chunk = json.loads(line)

                yield chunk["choices"][0]["delta"].get("content", "")

    except Exception as e:
        print(f"[chat_stream_generator] call model exception: {e}")
        yield "[model server error]"


def chat(
    messages: List[Dict[str, str]], endpoint: str, stop=None, temperature: float = 0.7
) -> str:
    """Function to get a single chat response from the backend."""

    payload = {"model": MODEL_NAME, "messages": messages, "temperature": temperature}
    if stop:
        payload["stop"] = stop

    try:
        with requests.post(
            BACKEND_HOST + endpoint,
            json=payload,
            headers=HEADERS,
            timeout=60 * 10,
        ) as response:
            if response.status_code != 200:
                return f"[model server error]. response text: {response.text}"

            model_response = response.json()
            return model_response["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"[chat] call model exception: {e}")
        return "[model server error]"
