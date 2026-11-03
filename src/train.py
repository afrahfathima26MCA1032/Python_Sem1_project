"""
train.py - trains and saves every model.

Run it with:   python3 -m src.train

It does three things:
  1. baseline()     compare Logistic Regression with Random Forest  -> outputs/run_summary.json
  2. calibration()  check the probabilities are honest and fix them -> outputs/calibration_summary.json
  3. app_model()    train the model the app uses                    -> models/model.joblib

Each function returns two things:
  - results: the numbers that are saved to a file
  - objects: the trained models and test data, used to draw charts in notebook 2
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import prep

RANDOM_SEED = 42                  # the same seed every time -> the same results every time
NUMBER_OF_TREES = 100
OUTPUTS_FOLDER = prep.PROJECT_FOLDER / "outputs"
MODELS_FOLDER = prep.PROJECT_FOLDER / "models"


def make_forest(min_loans_per_leaf, max_depth=None):
    """A Random Forest that gives extra weight to defaults (they are only 18% of loans)."""
    return RandomForestClassifier(n_estimators=NUMBER_OF_TREES,
                                  class_weight="balanced",
                                  min_samples_leaf=min_loans_per_leaf,
                                  max_depth=max_depth,
                                  n_jobs=-1,
                                  random_state=RANDOM_SEED)


def split(test_share):
    """Split the clean data into training loans and test loans (keeping the 82/18 ratio)."""
    loans = prep.load_clean()
    X = loans.drop(columns=["default"])
    y = loans["default"]
    return train_test_split(X, y, test_size=test_share, stratify=y, random_state=RANDOM_SEED)


def split_train_calibration_test():
    """60% to train, 20% to calibrate, 20% to test. The encoder learns from training loans only."""
    X_train, X_rest, y_train, y_rest = split(test_share=0.4)
    X_calib, X_test, y_calib, y_test = train_test_split(X_rest, y_rest, test_size=0.5,
                                                        stratify=y_rest, random_state=RANDOM_SEED)
    encoder = prep.fit_encoder(X_train, y_train)
    X_train = prep.encode(X_train, encoder)
    X_calib = prep.encode(X_calib, encoder)
    X_test = prep.encode(X_test, encoder)
    return encoder, X_train, y_train, X_calib, y_calib, X_test, y_test


def calibration_error(y_true, probabilities):
    """
    Expected calibration error (ECE): the average gap between
    what the model says and what really happened.

    Loans are put in 10 groups by predicted risk (0-10%, 10-20%, ...).
    In each group we compare the average prediction with the real default rate.
    """
    table = pd.DataFrame({"predicted": probabilities, "actual": np.asarray(y_true)})
    table["group"] = pd.cut(probabilities, np.linspace(0, 1, 11), include_lowest=True)

    total_error = 0.0
    for _, group in table.groupby("group", observed=True):
        gap = abs(group["actual"].mean() - group["predicted"].mean())
        share_of_loans = len(group) / len(table)
        total_error = total_error + share_of_loans * gap
    return float(total_error)


def baseline():
    """Step 1: a simple model (Logistic Regression) against a stronger one (Random Forest)."""
    X_train, X_test, y_train, y_test = split(test_share=0.2)
    encoder = prep.fit_encoder(X_train, y_train)
    X_train = prep.encode(X_train, encoder)
    X_test = prep.encode(X_test, encoder)

    # Logistic Regression needs the numbers on the same scale; a Random Forest does not.
    scaler = StandardScaler().fit(X_train)
    logistic = LogisticRegression(class_weight="balanced", max_iter=1000,
                                  random_state=RANDOM_SEED, n_jobs=-1)
    logistic.fit(scaler.transform(X_train), y_train)

    forest = make_forest(min_loans_per_leaf=5)
    forest.fit(X_train, y_train)

    # chance of default for every test loan
    logistic_probability = logistic.predict_proba(scaler.transform(X_test))[:, 1]
    forest_probability = forest.predict_proba(X_test)[:, 1]

    results_by_model = {
        "Logistic Regression": {"roc_auc": round(roc_auc_score(y_test, logistic_probability), 4),
                                "pr_auc": round(average_precision_score(y_test, logistic_probability), 4)},
        "Random Forest": {"roc_auc": round(roc_auc_score(y_test, forest_probability), 4),
                          "pr_auc": round(average_precision_score(y_test, forest_probability), 4)},
    }

    loans = prep.load_clean()
    results = {
        "raw_rows": 899164,
        "raw_cols": 27,
        "clean_rows": len(loans),
        "clean_cols": loans.shape[1],
        "n_features_model": X_train.shape[1],
        "default_rate": round(float(loans["default"].mean()), 4),
        "dropped_leaky_columns": prep.LEAKY_COLUMNS + ["Term"],
        "results": results_by_model,
    }
    (OUTPUTS_FOLDER / "run_summary.json").write_text(json.dumps(results, indent=2))

    objects = {"logistic": logistic, "forest": forest, "scaler": scaler,
               "X_test": X_test, "y_test": y_test}
    return results, objects


def calibration():
    """Step 2: are the predicted probabilities honest? Fix them on loans the forest never saw."""
    encoder, X_train, y_train, X_calib, y_calib, X_test, y_test = split_train_calibration_test()

    forest = make_forest(min_loans_per_leaf=5)
    forest.fit(X_train, y_train)
    before = forest.predict_proba(X_test)[:, 1]

    # Try both standard correction methods. The forest itself is frozen (not retrained).
    isotonic = CalibratedClassifierCV(FrozenEstimator(forest), method="isotonic").fit(X_calib, y_calib)
    sigmoid = CalibratedClassifierCV(FrozenEstimator(forest), method="sigmoid").fit(X_calib, y_calib)
    after_isotonic = isotonic.predict_proba(X_test)[:, 1]
    after_sigmoid = sigmoid.predict_proba(X_test)[:, 1]

    # keep whichever method has the smaller error
    if calibration_error(y_test, after_isotonic) <= calibration_error(y_test, after_sigmoid):
        chosen_method = "Isotonic"
        after = after_isotonic
    else:
        chosen_method = "Sigmoid"
        after = after_sigmoid

    results = {
        "chosen_method": chosen_method,
        "roc_auc_before": roc_auc_score(y_test, before),
        "roc_auc_after": roc_auc_score(y_test, after),
        "ece_before": calibration_error(y_test, before),
        "ece_after": calibration_error(y_test, after),
        "ece_sigmoid": calibration_error(y_test, after_sigmoid),
        "brier_before": brier_score_loss(y_test, before),
        "brier_after": brier_score_loss(y_test, after),
    }
    (OUTPUTS_FOLDER / "calibration_summary.json").write_text(json.dumps(results, indent=2))

    objects = {"before": before, "after": after, "y_test": y_test}
    return results, objects


def app_model():
    """
    Step 3: the model the app uses.
    Its trees are kept smaller (depth 20) so the SHAP explanations are about 8x faster.
    """
    encoder, X_train, y_train, X_calib, y_calib, X_test, y_test = split_train_calibration_test()

    forest = make_forest(min_loans_per_leaf=50, max_depth=20)
    forest.fit(X_train, y_train)
    calibrated = CalibratedClassifierCV(FrozenEstimator(forest), method="isotonic").fit(X_calib, y_calib)

    probability = calibrated.predict_proba(X_test)[:, 1]
    predicted_default = forest.predict(X_test)
    metrics = {
        "roc_auc": roc_auc_score(y_test, probability),
        "recall": recall_score(y_test, predicted_default),
        "precision": precision_score(y_test, predicted_default),
        "ece": calibration_error(y_test, probability),
    }

    # save everything the app needs in one file
    MODELS_FOLDER.mkdir(exist_ok=True)
    model_file = {"model": forest, "calibrated": calibrated, "encoder": encoder, "metrics": metrics}
    joblib.dump(model_file, MODELS_FOLDER / "model.joblib", compress=3)

    objects = {"forest": forest, "calibrated": calibrated, "X_test": X_test, "y_test": y_test}
    return metrics, objects


if __name__ == "__main__":
    print("baseline:", baseline()[0])
    print("calibration:", calibration()[0])
    print("app model:", app_model()[0])
