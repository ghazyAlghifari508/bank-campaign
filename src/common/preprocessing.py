import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


NUMERIC_FEATURES = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
BINARY_FEATURES = ["default", "housing", "loan"]
NOMINAL_FEATURES = ["job", "marital", "education", "contact", "month", "poutcome"]
OUTLIER_FEATURES = ["balance", "duration", "campaign", "pdays", "previous"]


class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns or OUTLIER_FEATURES

    def fit(self, X, y=None):
        self.bounds_ = {}
        for column in self.columns:
            q1 = X[column].quantile(0.25)
            q3 = X[column].quantile(0.75)
            iqr = q3 - q1
            self.bounds_[column] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        return self

    def transform(self, X):
        X = X.copy()
        for column, (lower, upper) in self.bounds_.items():
            X[column] = X[column].clip(lower=lower, upper=upper)
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


def build_knn_nn_preprocessor():
    encoder = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "bin",
                OrdinalEncoder(
                    categories=[["no", "yes"]] * len(BINARY_FEATURES),
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                BINARY_FEATURES,
            ),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), NOMINAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("cap_outliers", OutlierCapper()),
            ("encode_scale", encoder),
        ]
    )


def build_svm_preprocessor():
    categorical_features = BINARY_FEATURES + NOMINAL_FEATURES
    encoder = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("cap_outliers", OutlierCapper()),
            ("encode", encoder),
            ("scale", StandardScaler()),
        ]
    )


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=["deposit"])
    y = (df["deposit"] == "yes").astype(int)
    return X, y
