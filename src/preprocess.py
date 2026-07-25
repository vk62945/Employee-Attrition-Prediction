import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder
)
from sklearn.compose import ColumnTransformer


def prepare_data(df : pd.DataFrame):
    """
    Prepare dataset before training.
    """
    df = df.copy()
    # Drop unnecessary columns
    columns_to_drop = [
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours"
    ]

    df.drop(
        columns = columns_to_drop,
        inplace = True
    )

    # Encode target column
    label_encoder = LabelEncoder()

    df["Attrition"] = label_encoder.fit_transform(
        df["Attrition"]
    )

    return df

def get_feature_columns(x: pd.DataFrame):
    numerical_columns = x.select_dtypes(
        include=["int64"]
    ).columns.tolist()

    categorical_columns = x.select_dtypes(
        include=["object"]
    ).columns.tolist()

    return numerical_columns, categorical_columns

def build_preprocessor(
    numerical_columns,
    categorical_columns
):

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
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore"
                ),
                categorical_columns
            )
        ]
    )

    return preprocessor