from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
import pandas as pd
from src.actions.transform_data import FEATURE_SCHEMA

def map_normalize(X: pd.DataFrame) -> pd.DataFrame:
    return X.apply(lambda col: col.map(
                        lambda v: 1 if v == 1 or v == 1.0
                        else (0 if v == 0 or v == 0.0 else "UNKNOWN")
                    ))

def map_unknown(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype(str).where(X.astype(str).isin(["0", "1"]), "UNKNOWN")

def preprocess_fit(X : pd.DataFrame):

    mean_cols = FEATURE_SCHEMA["mean"]
    zero_cols = FEATURE_SCHEMA["zero"]
    cat_cols = FEATURE_SCHEMA["binary"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("mean", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), mean_cols),

            ("zero", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler())
            ]), zero_cols),
            ("cat", Pipeline([
                ("normalize", FunctionTransformer(map_normalize)),
                ("map_unknown", FunctionTransformer(map_unknown)),
                ("onehot", OneHotEncoder(
                    categories=[["0", "1", "UNKNOWN"]] * len(cat_cols),
                    handle_unknown="ignore",
                    sparse_output=False
                ))
            ]), cat_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False        
    )
    preprocessor.set_output(transform="pandas")

    X_imp = preprocessor.fit_transform(X)
    return X_imp, preprocessor