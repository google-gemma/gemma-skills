import os
from transformers import AutoTokenizer
from google.cloud import aiplatform

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")
ENDPOINT_ID = os.environ.get("GOOGLE_CLOUD_ENDPOINT_ID")

MODEL_ID = "google/gemma-4-31B-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def predict_gemma(project: str, endpoint_id: str, prompt: str, location: str = "us-central1"):
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

question = input("User: ")
messages = [
    {"role": "user", "content": question}
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

predict_gemma(project=PROJECT_ID, location=LOCATION, endpoint_id=ENDPOINT_ID, prompt=prompt)