import joblib
import os
import numpy as np

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "model", "url_model.pkl"
)

# Known safe domains whitelist
SAFE_DOMAINS = {
    "google.com", "youtube.com", "facebook.com",
    "wikipedia.org", "twitter.com", "instagram.com",
    "linkedin.com", "github.com", "microsoft.com",
    "apple.com", "amazon.com", "netflix.com",
    "reddit.com", "stackoverflow.com", "gmail.com"
}

try:
    rf_model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    rf_model = None
    print(f"Warning: Model not found. Run train.py first.")

def get_domain(url: str) -> str:
    url = url.strip()
    if url.startswith("https://"):
        url = url[8:]
    elif url.startswith("http://"):
        url = url[7:]
    return url.split("/")[0].replace("www.", "")

def predict_url(extracted_features: dict) -> tuple:
    if rf_model is None:
        return "Error: Model not loaded", 0.0

    feature_values = list(extracted_features.values())
    X_input = np.array(feature_values).reshape(1, -1)

    probabilities = rf_model.predict_proba(X_input)[0]
    prediction_code = rf_model.predict(X_input)[0]

    if prediction_code == 0:
        return "Safe", float(probabilities[0])
    else:
        return "Malicious", float(probabilities[1])

def predict_url_full(url: str, extracted_features: dict) -> tuple:
    """
    Full prediction with whitelist check first
    Returns: (prediction, confidence, source)
    """
    # Check whitelist first
    domain = get_domain(url)
    if domain in SAFE_DOMAINS:
        return "Safe", 0.99, "Whitelist"

    # Then use ML model
    prediction, confidence = predict_url(extracted_features)
    return prediction, confidence, "ML"