import gradio as gr
from transformers import pipeline, TextIteratorStreamer, GenerationConfig
from threading import Thread

# Load the pipeline
# Replace "google/gemma-4-E2B-it" with other available models
model_id = "google/gemma-4-E2B-it"

pipe = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
    dtype="auto",
)

def chat(message, history):
    messages = []

    # Add conversation history
    for msg in history:
        role = msg["role"]

        # Extract text from the content list (e.g. [{'text': 'hello', 'type': 'text'}])
        if isinstance(msg["content"], list):
            content_text = "".join([item["text"] for item in msg["content"] if item["type"] == "text"])
        else:
            content_text = msg["content"]

        messages.append({"role": role, "content": content_text})

    # Add current user message
    messages.append({"role": "user", "content": message})

    streamer = TextIteratorStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)
    config = GenerationConfig(max_new_tokens=256)
    thread = Thread(target=pipe, args=(messages,), kwargs=dict(
        generation_config=config,
        streamer=streamer
    ))
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