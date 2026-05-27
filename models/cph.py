import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler


def fit_and_evaluate_cph(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    continuous_cols: list[str] | None = None,
    penalizer: float = 0.1
) -> dict:
    """
    Fit a Cox proportional hazards model and evaluate validation C-index.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training predictors.
    y_train : pd.DataFrame
        Training targets with columns: duration, event.
    X_val : pd.DataFrame
        Validation predictors.
    y_val : pd.DataFrame
        Validation targets with columns: duration, event.
    continuous_cols : list[str] | None
        Columns to standardize. If None, no standardization is applied.
    penalizer : float
        L2 penalization strength for CoxPHFitter.

    Returns
    -------
    dict
        Dictionary containing model name, penalizer, validation C-index,
        partial hazards, scaler, and number of features.
    """
    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()

    # Standardize only selected continuous columns using training data statistics
    if continuous_cols is not None and len(continuous_cols) > 0:
        scaler = StandardScaler()

        X_train_processed[continuous_cols] = scaler.fit_transform(
            X_train_processed[continuous_cols]
        )
        X_val_processed[continuous_cols] = scaler.transform(
            X_val_processed[continuous_cols]
        )

    train_df = X_train_processed.copy()
    train_df["duration"] = y_train["duration"].values
    train_df["event"] = y_train["event"].values

    val_df = X_val_processed.copy()
    val_df["duration"] = y_val["duration"].values
    val_df["event"] = y_val["event"].values

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(train_df, duration_col="duration", event_col="event")

    partial_hazards = cph.predict_partial_hazard(X_val_processed)

    c_index = concordance_index(
        y_val["duration"],
        -partial_hazards.values.ravel(),
        y_val["event"]
    )

    results = {
    "model": "CoxPH",
    "val_c_index": float(c_index),
    "partial_hazards": partial_hazards,
    "fitted_model": cph,
    "scaler": scaler,
    "continuous_cols": continuous_cols,
    "penalizer": penalizer
}

    print(results)
    return results