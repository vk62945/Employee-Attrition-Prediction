import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from config import(
    RAW_DATA_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    MODEL_DIR,
    RANDOM_STATE
)

from utils import(
    load_data,
    save_model,
    create_directory
)

from preprocess import(
    prepare_data,
    get_feature_columns,
    build_preprocessor
)

def train_model():
    df = load_data(RAW_DATA_PATH)

    df = prepare_data(df)

    x = df.drop("Attrition", axis = 1)
    y = df["Attrition"]

    numerical_columns, categorical_columns = get_feature_columns(x)

    preprocessor = build_preprocessor(
        numerical_columns,
        categorical_columns
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    pipeline = Pipeline([
        ("Preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                random_state=RANDOM_STATE
            )
        )
    ])

    pipeline.fit(
        x_train,
        y_train
    )

    y_pred = pipeline.predict(x_test)

    print("\nModel Evaluation")
    print("-" * 40)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test,y_pred):.4f}")
    print(f"Recall: {recall_score(y_test,y_pred):.4f}")
    print(f"F1_score: {f1_score(y_test,y_pred):.4f}")
    print(f"ROC_AUC: {roc_auc_score(y_test,y_pred):.4f}")

    create_directory(MODEL_DIR)

    save_model(
        pipeline,
        MODEL_PATH
    )

    print("\nModel saved successfully")

if __name__ == "__main__":
    train_model()
