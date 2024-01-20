import os
from typing import List

from common.call_llm import chat_stream_generator

CHAT_ENDPOINT = os.environ.get("CHAT_ENDPOINT")


def generate_chat(
    input_text: str, history: List[List[str]], endpoint: str = CHAT_ENDPOINT
):
    """Generates chat responses and updates the chat history."""

    input_text = input_text or "你好"
    history = (history or [])[-5:]  # Keep the last 5 messages in history

    messages = []
    for message, answer in history:
        messages.append({"role": "user", "content": message})
        if answer:
            messages.append({"role": "assistant", "content": answer})

    # append latest message
    stream_response = chat_stream_generator(
        messages=messages, endpoint=endpoint, temperature=0.2
    )
    for character in stream_response:
        history[-1][1] += character
        yield None, history
    else:
        yield None, history
