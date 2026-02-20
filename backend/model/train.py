import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from core.feature_extractor import extract_features

def normalize_url(url: str) -> str:
    url = str(url).strip()
    if url.startswith("https://"):
        url = url[8:]
    elif url.startswith("http://"):
        url = url[7:]
    return url

def train_and_save_model():
    print("Loading dataset...")
    df = pd.read_csv(
        "backend/model/malicious_phish.csv",
        on_bad_lines='skip'
    )
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
    for url in df['url']:
        f = extract_features(url)
        features.append(list(f.values()))

    X = np.array(features)
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
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

    joblib.dump(model, "backend/model/url_model.pkl")
    print("Model saved!")

train_and_save_model()