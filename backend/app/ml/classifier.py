"""XGBoost Root Cause Classifier + SHAP Explainability."""

import time
import os
import numpy as np
import joblib
import shap
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from app.models.schemas import (
    FeatureVector, ClassificationResult, RootCause, ShapExplanation,
)
import structlog

logger = structlog.get_logger()

MODEL_DIR = Path(__file__).parent / "model_artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Categorical feature columns and their known categories.
#
# Every value the gateway can put in these fields needs a code here. `OrdinalEncoder` is
# configured with `unknown_value=-1`, so anything missing from a list is not an error — it
# encodes as "unrecognised" and shares that single code with every other unseen value.
#
# The lists these replaced were missing `payment_failed` and `limit_exceeded` from the
# reasons and `payment_authorization` from the steps. Those are not edge cases: in the
# failures the simulator emits, `payment_failed` is the single most common reason (it appears
# under all seven causes, which is exactly why it is there) and `payment_authorization` is
# the most common step. So the three highest-frequency values in the dataset were all being
# collapsed into one meaningless code, quietly, and the model was left to classify on what
# remained. `tests/test_classifier.py` now asserts these cover `app.sim.emission`, which is
# the only place the observable vocabulary is defined.
CATEGORICAL_FEATURES = {
    "error_source": ["customer", "gateway", "business", "razorpay"],
    "error_step": [
        "payment_initiation", "payment_authentication", "payment_authorization",
        "payment_processing", "payment_capture",
    ],
    "error_reason": [
        "payment_failed", "gateway_technical_error", "network_error", "timeout",
        "authentication_failed", "payment_cancelled", "insufficient_funds",
        "card_blocked", "invalid_card", "mandate_expired", "upi_psp_error",
        "bank_not_enabled", "limit_exceeded", "other",
    ],
    "payment_method": ["upi", "card", "netbanking", "wallet"],
    "amount_bucket": ["micro", "small", "medium", "large", "premium"],
}

NUMERICAL_FEATURES = [
    "hour_of_day", "day_of_week",
    "bank_success_rate_1h", "bank_failure_count_1h",
    "method_success_rate_1h",
    "customer_success_rate_30d", "customer_failure_count_7d",
    "customer_recovery_response",
]

BOOLEAN_FEATURES = [
    "is_month_end", "is_salary_window", "is_maintenance_window",
    "bank_is_in_downtime",
]

ALL_FEATURE_NAMES = (
    list(CATEGORICAL_FEATURES.keys()) + NUMERICAL_FEATURES + BOOLEAN_FEATURES
)


class RootCauseClassifier:
    """XGBoost multi-class classifier for payment failure root causes.

    17 features across four families — the payment's own error fields, the timing, the bank's
    recent health, and the customer's history. Trained by `app.ml.train`, which labels from
    the simulator's latent cause rather than from anything the model previously wrote, and
    reports the accuracy against the ceiling available from the error fields alone.

    Every prediction carries a SHAP attribution, and that is where the time goes: the model
    itself answers in about 0.7ms, the TreeExplainer takes it to roughly 15ms end to end.
    Worth paying — a root cause with no reason attached cannot be argued with by the ops
    person it is shown to — but it is 15ms, not the sub-10 this docstring used to claim.
    """

    def __init__(self):
        self.model: XGBClassifier | None = None
        self.label_encoder: LabelEncoder | None = None
        self.ordinal_encoder: OrdinalEncoder | None = None
        self.explainer: shap.TreeExplainer | None = None
        self._loaded = False

    def load(self):
        """Load trained model from disk."""
        model_path = MODEL_DIR / "xgb_model_v1.pkl"
        if not model_path.exists():
            logger.warning("classifier.no_model", path=str(model_path))
            return False

        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(MODEL_DIR / "label_encoder_v1.pkl")
        self.ordinal_encoder = joblib.load(MODEL_DIR / "ordinal_encoder_v1.pkl")
        self.explainer = shap.TreeExplainer(self.model)
        self._loaded = True
        logger.info("classifier.loaded")
        return True

    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, features: FeatureVector) -> ClassificationResult:
        """Classify root cause with SHAP explanation."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Train first via /api/v1/model/train")

        start = time.perf_counter()

        # Prepare feature array
        X = self._prepare_features(features)

        # Predict
        proba = self.model.predict_proba(X)[0]
        predicted_idx = proba.argmax()
        predicted_class = self.label_encoder.inverse_transform([predicted_idx])[0]
        confidence = float(proba.max())

        # SHAP explanations
        shap_values = self.explainer.shap_values(X)
        explanations = self._format_shap(
            shap_values, X, predicted_idx, features
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # All class probabilities
        all_probs = {
            self.label_encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(proba)
        }

        # Human-readable summary
        top_features = sorted(explanations, key=lambda x: abs(x.shap_value), reverse=True)[:3]
        summary_parts = []
        for exp in top_features:
            direction = "supports" if exp.shap_value < 0 else "against"
            summary_parts.append(
                f"{exp.feature}={exp.value} ({direction} {predicted_class})"
            )
        summary = f"Classified as {predicted_class} because: {'; '.join(summary_parts)}"

        return ClassificationResult(
            root_cause=RootCause(predicted_class),
            confidence=confidence,
            all_probabilities=all_probs,
            shap_explanations=explanations,
            explanation_summary=summary,
            inference_time_ms=round(elapsed_ms, 2),
        )

    def _prepare_features(self, features: FeatureVector) -> np.ndarray:
        """Convert FeatureVector to numpy array for model input."""
        data = features.to_dict()

        # Encode categorical features
        cat_values = []
        for col, categories in CATEGORICAL_FEATURES.items():
            val = data[col]
            cat_values.append([val])

        cat_encoded = self.ordinal_encoder.transform(
            np.array(cat_values).T
        )

        # Numerical features
        num_values = [data[col] for col in NUMERICAL_FEATURES]

        # Boolean features (as 0/1)
        bool_values = [int(data[col]) for col in BOOLEAN_FEATURES]

        X = np.concatenate([cat_encoded.flatten(), num_values, bool_values]).reshape(1, -1)
        return X

    def _format_shap(
        self, shap_values, X, predicted_idx, features: FeatureVector
    ) -> list[ShapExplanation]:
        """Format SHAP values into structured explanations."""
        data = features.to_dict()
        explanations = []

        # Safely extract 1D SHAP vector for the predicted class
        try:
            sv_array = np.array(shap_values)
            if sv_array.ndim == 3:
                # shape (n_samples, n_features, n_classes)
                sv = sv_array[0, :, predicted_idx].flatten()
            elif isinstance(shap_values, list):
                # list of (n_samples, n_features) arrays, one per class
                sv = np.array(shap_values[predicted_idx][0]).flatten()
            elif sv_array.ndim == 2:
                sv = sv_array[0].flatten()
            else:
                sv = sv_array.flatten()
        except Exception:
            sv = np.zeros(len(ALL_FEATURE_NAMES))

        predicted_class = self.label_encoder.inverse_transform([predicted_idx])[0]

        for i, fname in enumerate(ALL_FEATURE_NAMES):
            if i < len(sv):
                shap_val = float(sv[i]) if np.ndim(sv[i]) == 0 else float(np.mean(sv[i]))
            else:
                shap_val = 0.0
            direction = f"→ {predicted_class}" if shap_val < 0 else f"← {predicted_class}"

            explanations.append(ShapExplanation(
                feature=fname,
                value=data.get(fname, "N/A"),
                shap_value=round(shap_val, 4),
                direction=direction,
            ))

        # Sort by absolute SHAP value (most important first)
        explanations.sort(key=lambda x: abs(x.shap_value), reverse=True)
        return explanations[:10]  # Top 10 features

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """Fit on `X_train` and score on `X_test`, then save the artifacts.

        The split is the caller's, not this method's. It used to take one matrix and cut the
        last 20% off the end, which for simulated data is not a held-out sample of anything:
        rows within a batch share a world — the same outage windows, the same bank-hours, the
        same customers — so a cut inside a batch scores the model on payments whose
        circumstances it was fitted on. `app.ml.train` splits by seed, which puts whole
        independently generated batches on either side.

        Labels are encoded over the declared `RootCause` set rather than over whatever
        appeared in `y_train`. A cause missing from a training sample would otherwise shift
        every class index above it, and the saved encoder would disagree with the one the
        next run produces.
        """
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([c.value for c in RootCause])
        y_train_enc = self.label_encoder.transform(y_train)
        y_test_enc = self.label_encoder.transform(y_test)

        n_classes = len(self.label_encoder.classes_)
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=42,
        )
        self.model.fit(X_train, y_train_enc, eval_set=[(X_test, y_test_enc)], verbose=False)

        y_pred = self.model.predict(X_test)
        labels_idx = list(range(n_classes))
        report = classification_report(
            y_test_enc, y_pred,
            labels=labels_idx,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
        conf_matrix = confusion_matrix(y_test_enc, y_pred, labels=labels_idx).tolist()

        self.explainer = shap.TreeExplainer(self.model)
        self._loaded = True

        joblib.dump(self.model, MODEL_DIR / "xgb_model_v1.pkl")
        joblib.dump(self.label_encoder, MODEL_DIR / "label_encoder_v1.pkl")
        joblib.dump(self.ordinal_encoder, MODEL_DIR / "ordinal_encoder_v1.pkl")

        metrics = {
            "accuracy": round(report["accuracy"], 4),
            "macro_f1": round(report["macro avg"]["f1-score"], 4),
            "per_class": {
                cls: {
                    "precision": round(report[cls]["precision"], 4),
                    "recall": round(report[cls]["recall"], 4),
                    "f1": round(report[cls]["f1-score"], 4),
                    "support": int(report[cls]["support"]),
                }
                for cls in self.label_encoder.classes_
            },
            "confusion_matrix": conf_matrix,
            "class_order": list(self.label_encoder.classes_),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        }

        logger.info("classifier.trained",
                    accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"])
        return metrics


# Singleton
classifier = RootCauseClassifier()
