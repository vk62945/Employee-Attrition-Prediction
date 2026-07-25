import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder
)

from sklearn.compose import ColumnTransformer


def prepare_data(df):
    """
    Clean the dataset and prepare X and y.
    """

    # Drop unnecessary columns
    df = df.drop(
        columns=[
            "EmployeeNumber",
            "EmployeeCount",
            "Over18",
            "StandardHours"
        ]
    )

    # Encode target column
    le = LabelEncoder()
    df["Attrition"] = le.fit_transform(df["Attrition"])

    # Separate features and target
    x = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    return x, y


def build_preprocessor(x):
    """
    Create preprocessing pipeline.
    """

    # Numerical columns
    numerical_columns = x.select_dtypes(include=["int64"]).columns

    # Categorical columns
    categorical_columns = x.select_dtypes(include=["object"]).columns

    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_columns
            ),
            (
                "cat",
                OneHotEncoder(drop="first"),
                categorical_columns
            )
        ]
    )

    return preprocessor