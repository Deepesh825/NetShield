import sys
sys.path.append('D:/NetShield')

from backend.core.feature_extractor import extract_features
from backend.core.ml_model import predict_url_full

test_urls = [
    "https://google.com",
    "https://paypal-secure-login.suspicious-site.com/verify",
    "http://192.168.1.1/malware/download",
    "https://facebook.com",
    "https://totally-fake-paypal.ru/login",
    "https://vtop.vitbhopal.ac.in",
]

for url in test_urls:
    features = extract_features(url)
    prediction, confidence, source = predict_url_full(url, features)
    print(f"URL: {url}")
    print(f"Prediction: {prediction} ({confidence*100:.1f}%) via {source}")
    print("-" * 50)