import joblib
import os
import numpy as np

# Define the absolute path to where train.py saved the model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "url_model.pkl")

# Load the model into memory once when the application starts
try:
    rf_model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    rf_model = None
    print(f"Warning: Model file not found at {MODEL_PATH}. Run model/train.py first.")

def predict_url(extracted_features: dict) -> tuple:
    """
    Used for Tier 1 Local ML Prediction in `core/analyzer.py`,
    
    Takes the extracted features dictionary, passes it to the Random Forest,
    and returns a tuple: (Prediction_String, Confidence_Score)
    """
    if rf_model is None:
        return "Error: Model not loaded", 0.0

    # Convert the dictionary of features into a 2D numpy array (1 row, N columns)
    feature_values = list(extracted_features.values())
    X_input = np.array(feature_values).reshape(1, -1)

    # predict() returns an array like [0] or [1]
    prediction_code = rf_model.predict(X_input)[0]
    
    # predict_proba() returns probabilities for each class, e.g., [[0.2, 0.8]]
    # Index 0 is probability of being Benign (0), Index 1 is Malicious (1)
    probabilities = rf_model.predict_proba(X_input)[0]
    
    if prediction_code == 0:
        prediction_str = "Safe"
        confidence = probabilities[0]  # Confidence it is safe
    else:
        prediction_str = "Malicious"
        confidence = probabilities[1]  # Confidence it is malicious

    return prediction_str, float(confidence)