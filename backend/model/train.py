
import os

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

from backend.core.feature_extractor import extract_features, FEATURE_ORDER

def normalize_url(url: str) -> str:
    url = str(url).strip()
    if url.startswith("https://"):
        url = url[8:]
    elif url.startswith("http://"):
        url = url[7:]
    return url

def train_and_save_model():
    print("Loading dataset...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "malicious_phish.csv")

    df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
    print(f"Total URLs: {len(df)}")
    print(df['type'].value_counts())

    
    tranco_path = os.path.join(BASE_DIR, "tranco_top10k.csv")

    if os.path.exists(tranco_path):
        tranco_df = pd.read_csv(tranco_path, header=None)

        
        tranco_domains = tranco_df[1].tolist()

        tranco_extra = pd.DataFrame({
            "url": tranco_domains,
            "type": "benign"
        })

        df = pd.concat([df, tranco_extra], ignore_index=True)
        print(f"Added {len(tranco_domains)} real benign domains from Tranco.")
        
    extra_safe_urls = [
        "google.com",
        "facebook.com",
        "microsoft.com",
        "github.com",
        "amazon.com",
        "youtube.com",
        "wikipedia.org",
        "linkedin.com",
        "apple.com",
        "stackoverflow.com",
        "gmail.com",
        "instagram.com"
    ]

    extra_df = pd.DataFrame({
        "url": extra_safe_urls,
        "type": "benign"
    })

    df = pd.concat([df, extra_df], ignore_index=True)
    
    
    print(f"Total URLs: {len(df)}")
    print(df['type'].value_counts())

    # Normalize URLs before feature extraction
    df['url'] = df['url'].apply(normalize_url)

    # Binary label
    df['label'] = df['type'].apply(
        lambda x: 0 if x == 'benign' else 1
    )

    print("\nExtracting features...")
    features = []
    from backend.core.feature_extractor import FEATURE_ORDER

    for url in df['url']:
        f = extract_features(url)
        ordered_features = [f[key] for key in FEATURE_ORDER]
        features.append(ordered_features)

    X = np.array(features)
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=50,   
        max_depth=15,     
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    print("\nEvaluating...")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred,
        target_names=['Benign', 'Malicious']
    ))

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "url_model.pkl")

    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Model saved at {MODEL_SAVE_PATH}")
    
    print("Model classes:", model.classes_)

train_and_save_model()