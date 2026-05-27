# feature grouping functions (to split numerical and histogram variables)

import re # regular expression module, to uncover the feature naming pattern in the dataset
from collections import defaultdict # each new key starts automatically with empty list, avoiding Python Error
import pandas as pd
import numpy as np

def split_operational_feature_groups(operational_df: pd.DataFrame) -> dict:
    """
    Split operational columns into:
    - identifier/time columns
    - standalone numerical scalar features
    - flat histogram feature list
    - grouped histogram features by prefix
    """
    columns = operational_df.columns.tolist()

    id_columns = ["vehicle_id", "time_step"]
    feature_columns = [col for col in columns if col not in id_columns]

    grouped_by_prefix = defaultdict(list) # to group columns with the same prefix

    for col in feature_columns:
        # format: "start, one or more digits, underscore, one or more digits, end"
        match = re.match(r"^(\d+)_(\d+)$", col) 

        if not match:
            continue

        prefix, _ = match.groups()
        grouped_by_prefix[prefix].append(col)

    numerical_feature_cols = []
    histogram_feature_cols = []
    histogram_groups = {}

    for prefix, cols in grouped_by_prefix.items():
        sorted_cols = sorted(cols, key=lambda x: int(x.split("_")[1]))

        if len(sorted_cols) == 1 and sorted_cols[0].endswith("_0"):
            numerical_feature_cols.extend(sorted_cols)
        else:
            histogram_groups[prefix] = sorted_cols
            histogram_feature_cols.extend(sorted_cols)

    numerical_feature_cols = sorted(
        numerical_feature_cols,
        key=lambda x: int(x.split("_")[0])
    )

    histogram_feature_cols = sorted(
        histogram_feature_cols,
        key=lambda x: (int(x.split("_")[0]), int(x.split("_")[1]))
    )

    print("Identifier/time columns:", id_columns)
    print("Numerical scalar feature count:", len(numerical_feature_cols))
    print("Histogram feature count:", len(histogram_feature_cols))
    print("Histogram group count:", len(histogram_groups))

    return {
        "id_columns": id_columns,
        "numerical_feature_cols": numerical_feature_cols,
        "histogram_feature_cols": histogram_feature_cols,
        "histogram_groups": histogram_groups,
    }

def split_tabular_predictors_and_targets(
    aggregated_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Split aggregated tabular baseline dataframe into:
    - X: predictors
    - y_duration: survival duration
    - y_event: event indicator
    """
    X = aggregated_df.drop(columns=["duration", "event"])
    y_duration = aggregated_df["duration"].copy()
    y_event = aggregated_df["event"].copy()

    print("Predictor matrix shape:", X.shape)
    print("Duration shape:", y_duration.shape)
    print("Event shape:", y_event.shape)

    return X, y_duration, y_event

def identify_static_categorical_columns(X: pd.DataFrame) -> list[str]:
    """
    Identify static specification columns to be one-hot encoded.
    """
    static_categorical_cols = [col for col in X.columns if col.startswith("Spec_")]

    print("Static categorical columns:", static_categorical_cols)
    print("Static categorical column count:", len(static_categorical_cols))

    return static_categorical_cols

def fit_and_apply_one_hot_train(
    X_train: pd.DataFrame,
    static_categorical_cols: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """
    One-hot encode training data and return the encoded dataframe
    together with the resulting training columns
    """
    X_train_encoded = pd.get_dummies(
        X_train,
        columns=static_categorical_cols,
        drop_first=True,
        dtype = int
    )
    train_columns = X_train_encoded.columns.tolist()
    
    return X_train_encoded, train_columns

def apply_one_hot_to_other_split(
    X: pd.DataFrame,
    categorical_cols: list[str],
    train_columns: list[str]
) -> pd.DataFrame:
    """
    One-hot encode a non-training split and align columns to training.
    """
    X_enc = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype= int)
    X_enc = X_enc.reindex(columns=train_columns, fill_value=0)
    return X_enc

from sklearn.impute import SimpleImputer

def fit_and_apply_imputer_train(
    X: pd.DataFrame
) -> tuple[pd.DataFrame, SimpleImputer]:
    """
    Impute missing values in the numeric tabular predictor matrix using median imputation.
    Returns the imputed dataframe and the fitted imputer. Fit on the training set only, 
    use the imputer on the validation and test sets
    """
    imputer = SimpleImputer(strategy="median")

    X_imputed_array = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(
        X_imputed_array,
        columns=X.columns,
        index=X.index
    )

    print("Imputed predictor matrix shape:", X_imputed.shape)
    print("Remaining missing values:", X_imputed.isna().sum().sum())

    return X_imputed, imputer

def apply_tabular_imputer(
    X: pd.DataFrame,
    imputer: SimpleImputer
) -> pd.DataFrame:
    """
    Apply a previously fitted numeric imputer to a tabular predictor matrix.
    """
    X_imputed_array = imputer.transform(X)
    X_imputed = pd.DataFrame(
        X_imputed_array,
        columns=X.columns,
        index=X.index
    )

    print("Applied imputer to predictor matrix shape:", X_imputed.shape)
    print("Remaining missing values:", X_imputed.isna().sum().sum())

    return X_imputed

from sklearn.feature_selection import VarianceThreshold

def remove_near_zero_variance_features(
    X: pd.DataFrame,
    threshold: float = 1e-4
) -> tuple[pd.DataFrame, list[str], VarianceThreshold]:
    """
    Remove features with variance at or below the threshold.
    Returns:
    - reduced dataframe
    - removed column names
    - fitted selector
    """
    selector = VarianceThreshold(threshold=threshold)
    X_reduced_array = selector.fit_transform(X)

    kept_columns = X.columns[selector.get_support()].tolist()
    removed_columns = X.columns[~selector.get_support()].tolist()

    X_reduced = pd.DataFrame(
        X_reduced_array,
        columns=kept_columns,
        index=X.index
    )

    print("Original shape:", X.shape)
    print("Reduced shape after near-zero variance filtering:", X_reduced.shape)
    print("Removed feature count:", len(removed_columns))

    return X_reduced, removed_columns, selector


def remove_highly_correlated_features(
    X: pd.DataFrame,
    threshold: float = 0.90
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove highly correlated features using an an absolute correlation threshold.
    If two features are highly correlated, remove the one with the lower variance.
    If both have the same variance, remove the second one to keep the rule deterministic.
    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix containing only numeric columns.
    threshold : float, default=0.95
        Absolute correlation threshold above which one of two features is removed.
    Returns:
    - reduced dataframe
    - list of removed column names.
    """
    corr_matrix = X.corr().abs()
    variances = X.var()
    columns = corr_matrix.columns.tolist()
    to_drop = set()

    for i in range(len(columns)):
        col_i = columns[i]
        if col_i in to_drop:
            continue
        for j in range(i+1, len(columns)):
            col_j = columns[j]

            if col_j in to_drop:
                continue

            corr_value = corr_matrix.loc[col_i, col_j]

            if corr_value > threshold:
                var_i = variances[col_i]
                var_j = variances[col_j]

                if var_i < var_j:
                    to_drop.add(col_i)
                    break # no need to compare col_i further once marked for removal
                elif var_j < var_i:
                    to_drop.add(col_j)
                else:
                    # tie: remove the second one for reproducibility
                    to_drop.add(col_j)
    
    removed_columns = sorted(to_drop)
    X_reduced = X.drop(columns=removed_columns)

    print("Original shape:", X.shape)
    print("Reduced shape after correlation filtering:", X_reduced.shape)
    print("Removed correlated feature count:", len(removed_columns))

    return X_reduced, removed_columns

from sklearn.model_selection import train_test_split

def make_tabular_train_validation_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    validation_size: float = 0.2,
    random_state: int = 42,
    stratify_by_event: bool = True
):
    """
    Split one-row-per-vehicle tabular data into train and validation sets.
    """
    if stratify_by_event:
        stratify_values = y["event"]
    else:
        stratify_values = None

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify_values,
    )

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:", y_val.shape)

    print("Train event rate:", y_train["event"].mean())
    print("Validation event rate:", y_val["event"].mean())

    return X_train, X_val, y_train, y_val