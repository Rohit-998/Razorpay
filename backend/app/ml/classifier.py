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


# Categorical feature columns and their known categories
CATEGORICAL_FEATURES = {
    "error_source": ["customer", "gateway", "business", "razorpay"],
    "error_step": ["payment_authentication", "payment_initiation", "payment_capture", "payment_processing"],
    "error_reason": [
        "insufficient_funds", "gateway_technical_error", "authentication_failed",
        "payment_cancelled", "bank_not_enabled", "invalid_card", "card_blocked",
        "upi_psp_error", "network_error", "timeout", "mandate_expired", "other",
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
    """
    XGBoost multi-class classifier for payment failure root causes.
    
    - 17 features across 4 categories
    - SHAP TreeExplainer for per-prediction explanations
    - <10ms inference time
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

        # shap_values for multi-class: list of arrays, one per class
        if isinstance(shap_values, list):
            sv = shap_values[predicted_idx][0]
        else:
            sv = shap_values[0]

        predicted_class = self.label_encoder.inverse_transform([predicted_idx])[0]

        for i, fname in enumerate(ALL_FEATURE_NAMES):
            shap_val = float(sv[i]) if i < len(sv) else 0.0
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

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
        """Train the XGBoost model and save artifacts."""
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Time-series split: first 80% train, last 20% test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]

        # Train XGBoost
        n_classes = len(self.label_encoder.classes_)
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=42,
            use_label_encoder=False,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred = self.model.predict(X_test)
        labels_idx = list(range(len(self.label_encoder.classes_)))
        report = classification_report(
            y_test, y_pred,
            labels=labels_idx,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
        accuracy = report["accuracy"]
        macro_f1 = report["macro avg"]["f1-score"]
        conf_matrix = confusion_matrix(y_test, y_pred, labels=labels_idx).tolist()

        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        self._loaded = True

        # Save artifacts
        joblib.dump(self.model, MODEL_DIR / "xgb_model_v1.pkl")
        joblib.dump(self.label_encoder, MODEL_DIR / "label_encoder_v1.pkl")
        joblib.dump(self.ordinal_encoder, MODEL_DIR / "ordinal_encoder_v1.pkl")

        metrics = {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class": {
                cls: {
                    "precision": round(report[cls]["precision"], 4),
                    "recall": round(report[cls]["recall"], 4),
                    "f1": round(report[cls]["f1-score"], 4),
                    "support": report[cls]["support"],
                }
                for cls in self.label_encoder.classes_
            },
            "confusion_matrix": conf_matrix,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        logger.info("classifier.trained", accuracy=accuracy, macro_f1=macro_f1)
        return metrics


# Singleton
classifier = RootCauseClassifier()
