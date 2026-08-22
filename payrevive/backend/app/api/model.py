"""Model training and evaluation API."""
from fastapi import APIRouter
from app.ml.classifier import classifier, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, BOOLEAN_FEATURES, ALL_FEATURE_NAMES
from app.db.database import get_supabase
from app.cache.feature_store import feature_extractor
from app.models.schemas import FailedPayment, PaymentMethod, ErrorSource, FeatureVector
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path
import numpy as np
import joblib
from datetime import datetime
import structlog

logger = structlog.get_logger()
router = APIRouter()

MODEL_DIR = Path(__file__).parent.parent / "ml" / "model_artifacts"


@router.post("/model/train")
async def train_model():
    """Train XGBoost classifier on generated data."""
    db = get_supabase()

    # Fetch all payments with labels
    sessions = db.table("recovery_sessions").select("payment_id, root_cause").neq("root_cause", None).execute()
    if not sessions.data or len(sessions.data) < 20:
        return {"status": "error", "message": f"Need at least 20 labeled payments, have {len(sessions.data or [])}"}

    label_map = {s["payment_id"]: s["root_cause"] for s in sessions.data}
    payment_ids = list(label_map.keys())

    # Fetch payment data in chunks
    all_features = []
    all_labels = []

    for pid in payment_ids:
        p_result = db.table("payments").select("*").eq("payment_id", pid).single().execute()
        if not p_result.data:
            continue
        p = p_result.data

        try:
            payment = FailedPayment(
                payment_id=p["payment_id"],
                order_id=p.get("order_id"),
                amount=p["amount"],
                currency=p.get("currency", "INR"),
                method=PaymentMethod(p["method"]),
                bank=p.get("bank"),
                error_code=p.get("error_code", "UNKNOWN"),
                error_source=ErrorSource(p.get("error_source", "gateway")),
                error_step=p.get("error_step", "unknown"),
                error_reason=p.get("error_reason", "unknown"),
                error_description=p.get("error_description", ""),
                customer_contact=p.get("customer_contact"),
                customer_email=p.get("customer_email"),
                created_at=datetime.fromisoformat(p["created_at"]) if isinstance(p["created_at"], str) else p["created_at"],
            )

            features = await feature_extractor.extract(payment)
            all_features.append(features)
            all_labels.append(label_map[pid])
        except Exception as e:
            logger.warning("train.skip_payment", payment_id=pid, error=str(e))

    if len(all_features) < 20:
        return {"status": "error", "message": f"Only {len(all_features)} valid samples after extraction"}

    # Build feature matrix
    # Fit ordinal encoder on categorical features
    cat_data = []
    for f in all_features:
        d = f.to_dict()
        row = [d[col] for col in CATEGORICAL_FEATURES.keys()]
        cat_data.append(row)

    ordinal_encoder = OrdinalEncoder(
        categories=[cats for cats in CATEGORICAL_FEATURES.values()],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    cat_encoded = ordinal_encoder.fit_transform(cat_data)

    # Numerical + boolean features
    num_data = []
    for f in all_features:
        d = f.to_dict()
        num_row = [d[col] for col in NUMERICAL_FEATURES]
        bool_row = [int(d[col]) for col in BOOLEAN_FEATURES]
        num_data.append(num_row + bool_row)

    num_array = np.array(num_data)
    X = np.hstack([cat_encoded, num_array])
    y = np.array(all_labels)

    # Save ordinal encoder
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(ordinal_encoder, MODEL_DIR / "ordinal_encoder_v1.pkl")
    classifier.ordinal_encoder = ordinal_encoder

    # Train
    metrics = classifier.train(X, y, ALL_FEATURE_NAMES)

    logger.info("model.trained", accuracy=metrics["accuracy"], f1=metrics["macro_f1"])

    return {"status": "success", "metrics": metrics}


@router.get("/model/metrics")
async def get_model_metrics():
    """Get current model evaluation metrics."""
    if not classifier.is_loaded():
        return {"status": "not_trained", "message": "Model not trained yet"}
    return {"status": "loaded", "message": "Model is ready for predictions"}
