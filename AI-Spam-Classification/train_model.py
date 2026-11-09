
import pandas as pd
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent

# Load dataset
data = pd.read_csv(
    BASE_DIR / "data" / "sms.tsv",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# Convert labels
data["label"] = data["label"].map({"ham": 0, "spam": 1})
data.dropna(inplace=True)

X = data["message"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build ML pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )),
    ("classifier", MultinomialNB())
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save model
joblib.dump(pipeline, BASE_DIR / "model" / "spam_pipeline.joblib")

# Save evaluation metrics
metrics = {
    "accuracy": round(float(accuracy) * 100, 2),
    "total_samples": int(len(data)),
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "classification_report": classification_report(
        y_test, predictions,
        target_names=["Ham", "Spam"],
        output_dict=True
    )
}

with open(BASE_DIR / "model" / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Model training completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Dataset samples:", len(data))
print("Model saved successfully!")
