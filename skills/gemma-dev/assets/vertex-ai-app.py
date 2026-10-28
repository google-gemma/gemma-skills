"""Vertex AI chat application demo using Gemma and Hugging Face Transformers."""

import os

from google.cloud import aiplatform
import transformers

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")
ENDPOINT_ID = os.environ.get("GOOGLE_CLOUD_ENDPOINT_ID")

MODEL_ID = "google/gemma-4-31B-it"
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)


def predict_gemma(
    project: str,
    endpoint_id: str,
    prompt: str,
    location: str = "us-central1",
) -> None:
  """Sends a formatted prompt to a deployed Vertex AI Gemma endpoint.

  Args:
    project: The Google Cloud project ID.
    endpoint_id: The deployed Vertex AI endpoint ID.
    prompt: The chat-formatted prompt string to send to the model.
    location: The Google Cloud region hosting the endpoint.
  """
  # Initialize the Vertex AI client
  aiplatform.init(project=project, location=location)

  # Reference the deployed endpoint
  endpoint = aiplatform.Endpoint(endpoint_id)

  # Format the payload for Gemma 4
  instances = [{"prompt": prompt, "max_tokens": 1024}]

  # Generate prediction
  response = endpoint.predict(instances=instances)

  for prediction in response.predictions:
    print(prediction)


def main() -> None:
  """Prompts the user for input and queries the Vertex AI Gemma endpoint."""
  question = input("User: ")
  messages = [{"role": "user", "content": question}]
  formatted_prompt = tokenizer.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True
  )

  predict_gemma(
      project=PROJECT_ID,
      location=LOCATION,
      endpoint_id=ENDPOINT_ID,
      prompt=formatted_prompt,
  )


if __name__ == "__main__":
  main()
