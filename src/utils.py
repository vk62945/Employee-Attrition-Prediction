from pathlib import Path

import joblib
import pandas as pd

def load_data(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)

def create_directory(directory: Path):
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

def save_model(model, file_path: Path):
    joblib.dump(
        model,
        file_path
    )

def load_model(file_path: Path):
    return joblib.load(file_path)
