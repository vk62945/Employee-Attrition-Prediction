import pandas as pd
from src.config import MODEL_PATH
from src.utils import load_model

def load_prediction_model():
    """
    Load trained ML pipeline.
    """
    return load_model(MODEL_PATH)

def predict_attrition(input_data: dict):
    input_df = pd.DataFrame([input_data])

    model = load_prediction_model()
    prediction = model.predict(input_df)

    probability = model.predict_proba(input_df)

    prediction_label = (
        "Yes" if prediction[0] == 1 else "No"
    )

    confidence = probability.max() * 100

    return {
        "prediction": prediction_label,
        "confidence": round(confidence, 2)
    }
