# Vertex AI

Google Cloud's fully managed AI platform.

**Prerequisites:**
-   **Model Garden**: Search for "Gemma 4" in the Vertex AI Model Garden.
-   **Deploy**: Click "Deploy" to create an endpoint.
-   **Resource ID**: Select a model (e.g. gemma-4-31b-it).
-   **Endpoint access**: Configure your Endpoint Access based on your security requirements (Public or Private).
-   **API**: Note down your endpoint details to query the model via the **Vertex AI Prediction API**.

To use a Vertex AI endpoint, set the following environment variables:

- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud Project ID.
- `GOOGLE_CLOUD_LOCATION`: The region of your Vertex AI endpoint (e.g., `us-central1`).
- `GOOGLE_CLOUD_ENDPOINT_ID`: The ID of your Vertex AI endpoint.

**1. Install the SDK**
Run the following command to install the required client library:
```bash
pip install google-cloud-aiplatform
```

Essential for local development. It generates Application Default Credentials (ADC) so that client libraries (e.g., Python, Java) can automatically find and use your credentials.
```bash
gcloud auth application-default login
```

**2. Python Implementation**

Use Transformers AutoTokenizer to apply the chat tempalte.
```bash
pip install transformers jinja2
```

Create an `app.py` file to initialize the client and send an online prediction request:
```python
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
```

**Example Prompt:**
"Walk me through deploying Gemma 4 31B to a Vertex AI endpoint."
