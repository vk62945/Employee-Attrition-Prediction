from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    /"data"
    /"raw"
    /"WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    /"data"
    /"processed"
    /"processed_attrition.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    /"models"
)

MODEL_PATH = MODEL_DIR / "logistic_model.pkl"

PREPROCESSOR_PATH = (
    MODEL_DIR / "preprocessor.pkl"
)

RANDOM_STATE = 42
