"""Phase-1 model comparison: one untuned model per algorithm family.

    python -m src.models.train_models                 # all models
    python -m src.models.train_models --models lr,rf  # a subset

Each model is a full sklearn Pipeline, so scalers are fitted inside each CV fold.
Results: results/model_comparison.csv, results/cv_fold_scores.csv,
results/preds/<model>_student_test.parquet, artifacts/models/<model>.joblib
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .common import FEATURES, SEED, Experiment, run_experiment


class NearestArcBaseline(BaseEstimator, ClassifierMixin):
    """Non-learned reference point: score = 1 for the geometrically nearest
    candidate, 0 otherwise. Its log loss is not meaningful (hard 0/1 scores)."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = (X["dist_rank"].to_numpy() == 0).astype(float)
        return np.column_stack([1 - p, p])


EXPERIMENTS = {
    "baseline_nearest": Experiment(
        "baseline_nearest", "Rule-based baseline", "pick nearest candidate (dist_rank == 0)",
        lambda: NearestArcBaseline(), notes={"log_loss_na": True}),
    "logreg": Experiment(
        "logreg", "Linear / probabilistic", "StandardScaler + LogisticRegression(C=1.0, lbfgs, max_iter=2000)",
        lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))),
    "gnb": Experiment(
        "gnb", "Linear / probabilistic", "GaussianNB(var_smoothing=1e-9)",
        lambda: GaussianNB()),
    "knn": Experiment(
        "knn", "Instance-based", "StandardScaler + KNN(k=25, distance-weighted)",
        lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
        stride=10),  # KNN prediction cost grows with training size
    "random_forest": Experiment(
        "random_forest", "Ensemble (bagging)", "RandomForest(300 trees, min_samples_leaf=5)",
        lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
    "hist_gb": Experiment(
        "hist_gb", "Boosting", "HistGradientBoosting(lr=0.1, max_iter=300, early stopping)",
        lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, early_stopping=True,
                                               validation_fraction=0.1, random_state=SEED)),
    "mlp": Experiment(
        "mlp", "Neural network", "StandardScaler + MLP(64, 32), adam, early stopping",
        lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                                              max_iter=200, random_state=SEED))),
    "decision_tree": Experiment(
        "decision_tree", "Tree-based", "DecisionTree(unlimited depth, min_samples_leaf=1)",
        lambda: DecisionTreeClassifier(random_state=SEED)),
    "extra_trees": Experiment(
        "extra_trees", "Ensemble (randomised trees)", "ExtraTrees(300 trees, min_samples_leaf=5)",
        lambda: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
}

# Tuned HistGB (written by src/models/tune_hgb.py). Registered here so the same runner and
# src/models/oof.py can rebuild it: python -m src.models.oof --model hist_gb_tuned
_TUNED_INFO = Path("artifacts/hist_gb_tuned_params.json")
if _TUNED_INFO.exists():
    _tuned_params = json.loads(_TUNED_INFO.read_text())["params"]

    def _make_tuned():
        from .tune_hgb import make_hgb
        return make_hgb(_tuned_params)

    EXPERIMENTS["hist_gb_tuned"] = Experiment(
        "hist_gb_tuned", "Boosting (tuned)", "HistGB with Optuna-tuned parameters (artifacts/hist_gb_tuned_params.json)",
        _make_tuned)

ALIASES = {"lr": "logreg", "rf": "random_forest", "hgb": "hist_gb", "nb": "gnb", "base": "baseline_nearest",
           "dt": "decision_tree", "et": "extra_trees"}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="all")
    a = p.parse_args(argv)
    names = list(EXPERIMENTS) if a.models == "all" else [ALIASES.get(m, m) for m in a.models.split(",")]
    print(f"{len(FEATURES)} features; running: {', '.join(names)}")
    for n in names:
        print(f"\n== {n} ==")
        run_experiment(EXPERIMENTS[n])


if __name__ == "__main__":
    main()
