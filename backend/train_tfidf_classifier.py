"""Utility script to train a TF-IDF + Logistic Regression classifier."""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from typing import Dict, List, Tuple

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_DATA_PATH = "clean_training_data.json"
DEFAULT_OUTPUT_PATH = "checkpoints/tfidf_classifier.joblib"
DEFAULT_METRICS_PATH = "checkpoints/tfidf_metrics.json"


def load_dataset(path: str) -> Tuple[List[str], List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as data_file:
        raw_data = json.load(data_file)

    texts: List[str] = []
    labels: List[str] = []

    for item in raw_data:
        text = (item.get("text") or "").strip()
        label = item.get("category")
        if not text or not label:
            continue
        texts.append(text)
        labels.append(label)

    if not texts:
        raise ValueError("Dataset is empty after filtering. Provide a valid dataset.")

    return texts, labels


def build_pipeline() -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=25000,
        min_df=2,
        ngram_range=(1, 2),
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w[\w']+\b",
        stop_words="english",
    )

    classifier = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        verbose=0,
    )

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def train_model(data_path: str, model_path: str, metrics_path: str) -> Dict:
    texts, labels = load_dataset(data_path)

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=sorted(set(labels))).tolist()

    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    metrics_dir = os.path.dirname(metrics_path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    joblib.dump({
        "pipeline": pipeline,
        "categories": list(pipeline.classes_),
        "trained_at": timestamp,
    }, model_path)

    metrics_payload = {
        "data_path": data_path,
        "model_path": model_path,
        "trained_at": timestamp,
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": {
            "labels": sorted(set(labels)),
            "matrix": conf_matrix,
        },
        "n_train": len(x_train),
        "n_test": len(x_test),
    }

    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics_payload, metrics_file, indent=2)

    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a TF-IDF logistic regression classifier.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to the JSON training dataset.")
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_PATH, help="Destination path for the trained model (joblib)."
    )
    parser.add_argument(
        "--metrics", default=DEFAULT_METRICS_PATH, help="Destination path for the training metrics (JSON)."
    )

    args = parser.parse_args()

    print("🚀 Training TF-IDF classifier...")
    metrics = train_model(args.data, args.output, args.metrics)
    print(f"✅ Training complete. Accuracy: {metrics['accuracy']:.3f}")
    print(f"📦 Model saved to: {args.output}")
    print(f"📊 Metrics saved to: {args.metrics}")


if __name__ == "__main__":
    main()
