"""
Fit and evaluate two RSF variants:
(1) an untuned literature-based baseline and
(2) a tuned version based on a restricted randomized search.

The design follows a controlled comparison strategy. First, a baseline model
is trained using hyperparameters reported in prior work on the SCANIA dataset.
Second, a small local search is performed around that configuration in order
to allow limited adaptation to the present preprocessing pipeline.

This approach is preferred over a broad hyperparameter search because it is
more computationally feasible, more stable on limited hardware, and more
consistent with the goal of fair and reproducible baseline comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, RandomizedSearchCV
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored


def to_sksurv_y(y: pd.DataFrame) -> np.ndarray:
    """
    Convert target dataframe to scikit-survival structured array.
    """
    return np.array(
        [(bool(event), float(duration)) for event, duration in zip(y["event"], y["duration"])],
        dtype=[("event", "?"), ("duration", "<f8")]
    )


def fit_and_evaluate_rsf_baseline(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    random_state: int = 42,
) -> dict:
    """
    Fit and evaluate one untuned RSF baseline using literature-based hyperparameters.
    """
    y_train_struct = to_sksurv_y(y_train)
    y_val_struct = to_sksurv_y(y_val)

    rsf_baseline = RandomSurvivalForest(
        n_estimators=100,
        max_depth=30,
        min_samples_split=30,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=1,
        verbose=0
    )

    rsf_baseline.fit(X_train, y_train_struct)

    val_risk_scores = rsf_baseline.predict(X_val)

    val_c_index = concordance_index_censored(
        y_val_struct["event"],
        y_val_struct["duration"],
        val_risk_scores
    )[0]

    results = {
        "model": "RSF_baseline",
        "params": {
            "n_estimators": 100,
            "max_depth": 30,
            "min_samples_split": 30,
            "min_samples_leaf": 20,
            "max_features": "sqrt"
        },
        "val_c_index": float(val_c_index),
        "n_features": X_train.shape[1],
        "fitted_model": rsf_baseline
    }

    print("RSF baseline hyperparameters:")
    print(results["params"])
    print(f"Baseline validation C-index: {val_c_index:.4f}")

    return results


def fit_and_evaluate_rsf_with_random_search(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    random_state: int = 42,
    n_iter: int = 10,
    cv_splits: int = 3,
    verbose: int = 1
) -> dict:
    """
    Tune Random Survival Forest hyperparameters using a restricted randomized search,
    refit the best model on the full training set, and evaluate on the validation set.

    Balanced setup:
    - RSF n_jobs = 1  -> avoids heavy parallel tree-building memory load
    - RandomizedSearchCV n_jobs = 2 -> allows limited parallelism across CV fits
    """
    y_train_struct = to_sksurv_y(y_train)
    y_val_struct = to_sksurv_y(y_val)

    rsf = RandomSurvivalForest(
        random_state=random_state,
        n_jobs=1,
        verbose=0
    )

    param_distributions = {
        "n_estimators": [80, 100, 120],
        "max_depth": [20, 30, 40],
        "min_samples_split": [20, 30, 40],
        "min_samples_leaf": [10, 20, 30],
        "max_features": ["sqrt", "log2"]
    }

    cv = KFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )

    random_search = RandomizedSearchCV(
        estimator=rsf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        n_jobs=2,
        refit=True,
        random_state=random_state,
        verbose=verbose,
        error_score="raise"
    )

    random_search.fit(X_train, y_train_struct)

    best_params = random_search.best_params_
    best_cv_score = random_search.best_score_

    best_rsf = RandomSurvivalForest(
        **best_params,
        random_state=random_state,
        n_jobs=1,
        verbose=0
    )
    best_rsf.fit(X_train, y_train_struct)

    val_risk_scores = best_rsf.predict(X_val)

    val_c_index = concordance_index_censored(
        y_val_struct["event"],
        y_val_struct["duration"],
        val_risk_scores
    )[0]

    results = {
        "model": "RSF_tuned",
        "best_params": best_params,
        "best_cv_c_index": float(best_cv_score),
        "val_c_index": float(val_c_index),
        "n_features": X_train.shape[1],
        "fitted_model": best_rsf
    }

    print("Best hyperparameters found by randomized search:")
    print(best_params)
    print(f"Best CV C-index: {best_cv_score:.4f}")
    print(f"Tuned validation C-index: {val_c_index:.4f}")

    return results


def fit_and_evaluate_rsf_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    random_state: int = 42,
    n_iter: int = 10,
    cv_splits: int = 3,
    verbose: int = 1
) -> dict:
    """
    Fit and evaluate:
    1. One untuned literature-based RSF baseline
    2. One tuned RSF with restricted randomized search

    Returns both result dictionaries.
    """
    baseline_results = fit_and_evaluate_rsf_baseline(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        random_state=random_state
    )

    tuned_results = fit_and_evaluate_rsf_with_random_search(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        random_state=random_state,
        n_iter=n_iter,
        cv_splits=cv_splits,
        verbose=verbose
    )

    return {
        "baseline": baseline_results,
        "tuned": tuned_results
    }