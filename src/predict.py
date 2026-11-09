"""
predict.py
----------
Loads the trained model ONCE and exposes predict_message().
"""
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
# The saved model refers to "src.preprocess", so the project root must be importable
# even when this file is run directly (python src/predict.py).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import src.preprocess  # noqa: E402,F401
MODEL_FILE = ROOT / "model" / "spam_pipeline.joblib"
MAX_CHARS = 20000

_model = None


def load_model():
    """Load the model the first time it is needed, then reuse it."""
    global _model
    if _model is None:
        if not MODEL_FILE.exists():
            raise FileNotFoundError("Model not found. Run:  python src/train.py")
        _model = joblib.load(MODEL_FILE)
    return _model


def predict_message(text: str) -> dict:
    """Return the label and probabilities for one email."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Email text is empty.")
    text = text[:MAX_CHARS]
    model = load_model()
    classes = list(model.classes_)
    probs = model.predict_proba([text])[0]
    spam_prob = float(probs[classes.index("spam")])
    is_spam = spam_prob >= 0.5
    return {
        "label": "Spam" if is_spam else "Not Spam",
        "is_spam": is_spam,
        "spam_probability": round(spam_prob * 100, 2),
        "confidence": round(max(spam_prob, 1 - spam_prob) * 100, 2),
    }


if __name__ == "__main__":
    tests = [
        "Congratulations! You won a FREE prize. Click http://claim-now.biz to get your money NOW!!!",
        "Hi team, attaching the agenda for tomorrow's project review meeting. Please read before 10am.",
    ]
    for t in tests:
        print(t[:60], "->", predict_message(t))
