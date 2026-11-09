"""
train.py
--------
Trains the spam classifier and saves everything the website needs:
    model/spam_pipeline.joblib   (TF-IDF + Multinomial Naive Bayes)
    model/metrics.json           (accuracy, precision, recall, F1, confusion matrix, ...)
    static/img/confusion_matrix.png

Run from the project root:  python src/train.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # draw images without opening a window (needed on servers)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Make "src" importable no matter where the script is launched from
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.preprocess import clean_text  # noqa: E402

DATA_FILE = ROOT / "data" / "emails.csv"
MODEL_FILE = ROOT / "model" / "spam_pipeline.joblib"
METRICS_FILE = ROOT / "model" / "metrics.json"
MATRIX_IMG = ROOT / "static" / "img" / "confusion_matrix.png"
RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,   # our shared cleaning function
            stop_words="english",
            ngram_range=(1, 2),        # single words and word pairs
            min_df=2,                  # ignore words seen in only 1 email
            max_df=0.95,
            max_features=50000,
            sublinear_tf=True,
        )),
        ("nb", MultinomialNB(alpha=0.1)),
    ])


def top_spam_words(pipeline: Pipeline, n: int = 15) -> list:
    """Words that push an email most strongly towards SPAM."""
    vec = pipeline.named_steps["tfidf"]
    nb = pipeline.named_steps["nb"]
    classes = list(nb.classes_)
    diff = nb.feature_log_prob_[classes.index("spam")] - nb.feature_log_prob_[classes.index("ham")]
    names = np.array(vec.get_feature_names_out())
    return [w for w in names[np.argsort(diff)[::-1][:n]] if w not in ("urltoken", "numtoken", "emailtoken")]


def save_confusion_matrix(cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=150)
    ax.imshow(cm, cmap="Blues")
    labels = ["Not Spam", "Spam"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    MATRIX_IMG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(MATRIX_IMG)
    plt.close(fig)


def main() -> None:
    if not DATA_FILE.exists():
        sys.exit("data/emails.csv not found. Run:  python src/build_dataset.py")

    df = pd.read_csv(DATA_FILE).dropna(subset=["text", "label"])
    print(f"Loaded {len(df)} emails")
    print(df["label"].value_counts().to_string(), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"])

    model = build_pipeline()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, pos_label="spam"), 4),
        "recall": round(recall_score(y_test, y_pred, pos_label="spam"), 4),
        "f1_score": round(f1_score(y_test, y_pred, pos_label="spam"), 4),
        "confusion_matrix": {"true_negative": tn, "false_positive": fp,
                             "false_negative": fn, "true_positive": tp},
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "total_emails": int(len(df)),
        "spam_emails": int((df["label"] == "spam").sum()),
        "ham_emails": int((df["label"] == "ham").sum()),
        "top_spam_words": top_spam_words(model),
        "sklearn_version": sklearn.__version__,
        "model": "TF-IDF + Multinomial Naive Bayes",
    }

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE, compress=3)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(cm)

    print(f"Accuracy : {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}  (of emails flagged spam, how many really were)")
    print(f"Recall   : {metrics['recall']:.2%}  (of real spam, how much we caught)")
    print(f"F1-score : {metrics['f1_score']:.2%}")
    print("Confusion matrix [[TN FP] [FN TP]]:", cm.tolist())
    print("\nSaved model, metrics.json and confusion_matrix.png")


if __name__ == "__main__":
    main()
