"""Baseline models: Random Forest and XGBoost per-task classifiers."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class PerTaskClassifier:
    """Wrapper for training independent binary classifiers per Tox21 assay.

    Handles missing labels by training each task only on observed samples.
    Supports Random Forest and XGBoost backends.

    Args:
        model_type: Either "random_forest" or "xgboost".
        task_names: List of assay/task names.
        model_params: Dictionary of model hyperparameters.
        auto_pos_weight: If True, compute scale_pos_weight / class_weight
            dynamically from the training data.
    """

    def __init__(
        self,
        model_type: str,
        task_names: list[str],
        model_params: Optional[dict] = None,
        auto_pos_weight: bool = True,
    ) -> None:
        self.model_type = model_type
        self.task_names = task_names
        self.model_params = model_params or {}
        self.auto_pos_weight = auto_pos_weight
        self.models: dict[str, object] = {}

    def _create_model(self, pos_weight: Optional[float] = None) -> object:
        """Create a single classifier instance.

        Args:
            pos_weight: Ratio of negatives to positives for class weighting.

        Returns:
            Scikit-learn compatible classifier.
        """
        params = self.model_params.copy()

        if self.model_type == "random_forest":
            # class_weight="balanced" handles imbalance automatically
            return RandomForestClassifier(**params)

        elif self.model_type == "xgboost":
            # Set scale_pos_weight dynamically if requested
            if self.auto_pos_weight and pos_weight is not None:
                params["scale_pos_weight"] = pos_weight
            # Remove null/None scale_pos_weight so XGBoost uses default
            if params.get("scale_pos_weight") is None:
                params.pop("scale_pos_weight", None)
            # Remove early_stopping_rounds from constructor — passed to fit()
            params.pop("early_stopping_rounds", None)
            params.pop("eval_metric", None)
            return XGBClassifier(**params)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: Optional[int] = None,
    ) -> None:
        """Train one classifier per task on observed labels only.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            y: Label matrix, shape (n_samples, n_tasks). NaN = unobserved.
            X_val: Validation features for XGBoost early stopping.
            y_val: Validation labels for XGBoost early stopping.
            early_stopping_rounds: Rounds for XGBoost early stopping.
        """
        n_tasks = y.shape[1]
        assert n_tasks == len(self.task_names), (
            f"Label columns ({n_tasks}) != task names ({len(self.task_names)})"
        )

        for i, task_name in enumerate(self.task_names):
            logger.info(f"Training {self.model_type} for {task_name}...")

            # Mask to observed labels only
            mask = ~np.isnan(y[:, i])
            X_task = X[mask]
            y_task = y[mask, i].astype(int)

            n_pos = int(y_task.sum())
            n_neg = int(len(y_task) - n_pos)
            pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

            logger.info(
                f"  {task_name}: {len(y_task)} samples "
                f"({n_pos} pos, {n_neg} neg, pos_weight={pos_weight:.1f})"
            )

            model = self._create_model(pos_weight=pos_weight)

            # XGBoost with early stopping
            if (
                self.model_type == "xgboost"
                and X_val is not None
                and y_val is not None
                and early_stopping_rounds is not None
            ):
                val_mask = ~np.isnan(y_val[:, i])
                if val_mask.sum() > 0:
                    model.set_params(
                        early_stopping_rounds=early_stopping_rounds,
                        eval_metric="auc",
                    )
                    model.fit(
                        X_task,
                        y_task,
                        eval_set=[(X_val[val_mask], y_val[val_mask, i].astype(int))],
                        verbose=False,
                    )
                else:
                    model.fit(X_task, y_task)
            else:
                model.fit(X_task, y_task)

            self.models[task_name] = model

        logger.info(f"Trained {len(self.models)} per-task {self.model_type} models")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for all tasks.

        Args:
            X: Feature matrix, shape (n_samples, n_features).

        Returns:
            Probability matrix, shape (n_samples, n_tasks). Each column
            contains P(active) for that assay.
        """
        n_samples = X.shape[0]
        probs = np.full((n_samples, len(self.task_names)), np.nan)

        for i, task_name in enumerate(self.task_names):
            if task_name not in self.models:
                logger.warning(f"No model found for {task_name}, skipping")
                continue

            model = self.models[task_name]
            # predict_proba returns (n_samples, 2) — take P(class=1)
            probs[:, i] = model.predict_proba(X)[:, 1]

        return probs

    def save(self, path: str | Path) -> None:
        """Save all models to a pickle file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_type": self.model_type,
            "task_names": self.task_names,
            "model_params": self.model_params,
            "auto_pos_weight": self.auto_pos_weight,
            "models": self.models,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved {self.model_type} models to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "PerTaskClassifier":
        """Load models from a pickle file.

        Args:
            path: Path to the saved model file.

        Returns:
            PerTaskClassifier with loaded models.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        obj = cls(
            model_type=state["model_type"],
            task_names=state["task_names"],
            model_params=state["model_params"],
            auto_pos_weight=state["auto_pos_weight"],
        )
        obj.models = state["models"]

        logger.info(f"Loaded {obj.model_type} models from {path}")
        return obj
