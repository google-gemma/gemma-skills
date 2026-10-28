"""Gradio chat application demo using Gemma and Hugging Face Transformers."""

from collections.abc import Iterator, Mapping, Sequence
import threading
from typing import Any

import gradio as gr
import transformers

# Load the pipeline
# Replace "google/gemma-4-E2B-it" with other available models
model_id = "google/gemma-4-E2B-it"

pipe = transformers.pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
    dtype="auto",
)


def chat(message: str, history: Sequence[Mapping[str, Any]]) -> Iterator[str]:
  """Generates streaming chat responses for a Gradio ChatInterface.

  Args:
    message: The current user message.
    history: Previous conversation messages with role and content fields.

  Yields:
    Accumulated generated response text as tokens stream in.
  """
  messages = []

  # Add conversation history
  for msg in history:
    role = msg["role"]

    # Extract text from the content list
    # (e.g. [{'text': 'hello', 'type': 'text'}])
    if isinstance(msg["content"], list):
      content_text = "".join(
          [item["text"] for item in msg["content"] if item["type"] == "text"]
      )
    else:
      content_text = msg["content"]

    messages.append({"role": role, "content": content_text})

  # Add current user message
  messages.append({"role": "user", "content": message})

  streamer = transformers.TextIteratorStreamer(
      pipe.tokenizer, skip_prompt=True, skip_special_tokens=True
  )
  config = transformers.GenerationConfig(max_new_tokens=256)
  thread = threading.Thread(
      target=pipe,
      args=(messages,),
      kwargs=dict(generation_config=config, streamer=streamer),
  )
  thread.start()

  # Generate response
  generated_text = ""
  for new_text in streamer:
    generated_text += new_text
    yield generated_text


# Create the ChatInterface
demo = gr.ChatInterface(
    fn=chat,
    title="Gemma Chatbot",
    description="Ask Gemma anything!",
)

if __name__ == "__main__":
  demo.launch()
