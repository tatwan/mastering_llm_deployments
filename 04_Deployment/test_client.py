import requests

BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print("Health:", response.json())

# Single prediction
response = requests.post(
    f"{BASE_URL}/predict",
    json={"text": "This movie was fantastic!"}
)
print("Prediction:", response.json())

# Batch prediction
response = requests.post(
    f"{BASE_URL}/predict/batch",
    json={
        "texts": [
            "I love this product!",
            "Terrible experience, never again.",
            "It was okay."
        ]
    }
)
print("Batch:", response.json())