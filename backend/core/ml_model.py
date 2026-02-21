import joblib
import os
import numpy as np

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "model",
    "url_model.pkl"
)


SAFE_DOMAINS = {
    "google.com", "youtube.com", "facebook.com",
    "wikipedia.org", "twitter.com", "instagram.com",
    "linkedin.com", "github.com", "microsoft.com",
    "apple.com", "amazon.com", "netflix.com",
    "reddit.com", "stackoverflow.com", "gmail.com"
}


DOMAIN_MEMORY = {}


try:
    rf_model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    rf_model = None
    print("Warning: Model not found. Run train.py first.")



def get_domain(url: str) -> str:
    url = url.strip().lower()
    if url.startswith("https://"):
        url = url[8:]
    elif url.startswith("http://"):
        url = url[7:]
    return url.split("/")[0].replace("www.", "")



def predict_url(extracted_features: dict) -> tuple:
    if rf_model is None:
        return "Error", 0.0

    from backend.core import feature_extractor

    feature_values = [extracted_features[key] for key in feature_extractor.FEATURE_ORDER]
    X_input = np.array(feature_values).reshape(1, -1)

    # Get probability safely
    probs = rf_model.predict_proba(X_input)[0]
    malicious_index = list(rf_model.classes_).index(1)
    prob_malicious = float(probs[malicious_index])



    if prob_malicious >= 0.90:
        return "Malicious", prob_malicious

    elif prob_malicious >= 0.60:
        return "Suspicious", prob_malicious

    elif prob_malicious >= 0.30:
        return "Low Risk", 1 - prob_malicious

    else:
        return "Safe", 1 - prob_malicious



def predict_url_full(url: str, extracted_features: dict) -> tuple:
    domain = get_domain(url)


    for safe in SAFE_DOMAINS:
        if domain == safe or domain.endswith("." + safe):
            return "Safe", 0.99, ["Whitelisted domain"]


    if domain in DOMAIN_MEMORY:
        return DOMAIN_MEMORY[domain]


    prediction, confidence = predict_url(extracted_features)

    reasons = []

    if extracted_features.get("contains_suspicious_keyword"):
        reasons.append("Suspicious keyword detected")

    if extracted_features.get("risky_tld"):
        reasons.append("Risky TLD detected")

    if extracted_features.get("many_hyphens"):
        reasons.append("Multiple hyphens in domain")

    if extracted_features.get("has_ip"):
        reasons.append("IP address used instead of domain")

    if not reasons:
        reasons.append("No major risk indicators")

    result = (prediction, confidence, reasons)


    DOMAIN_MEMORY[domain] = result

    return result