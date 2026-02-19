import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import sys
import os

# Add parent directory to path so we can import feature_extractor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.feature_extractor import extract_features

# ── Step 1: Load Dataset ──────────────────────────────
print("Loading dataset...")
df = pd.read_csv("backend/model/malicious_phish.csv")
print(f"Total URLs: {len(df)}")
print(df['type'].value_counts())

# ── Step 2: Label Encoding ────────────────────────────
# Convert text labels to numbers
# benign = 0, everything else = 1 (malicious)
df['label'] = df['type'].apply(
    lambda x: 0 if x == 'benign' else 1
)

# ── Step 3: Feature Extraction ────────────────────────
print("\nExtracting features... (this takes a few minutes)")

features = []
for url in df['url']:
    f = extract_features(str(url))
    features.append(list(f.values()))

X = np.array(features)
y = df['label'].values

print(f"Features shape: {X.shape}")

# ── Step 4: Train/Test Split ──────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 80% train, 20% test
    random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ── Step 5: Train Model ───────────────────────────────
print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1           # uses all CPU cores
)
model.fit(X_train, y_train)

# ── Step 6: Evaluate ──────────────────────────────────
print("\nEvaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred,
    target_names=['Benign', 'Malicious']
))

# ── Step 7: Save Model ────────────────────────────────
model_path = "backend/model/url_model.pkl"
joblib.dump(model, model_path)
print(f"\nModel saved to {model_path}")