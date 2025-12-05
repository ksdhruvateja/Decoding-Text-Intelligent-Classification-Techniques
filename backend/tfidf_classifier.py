"""TF-IDF based text classifier used as a lightweight fallback model."""
from __future__ import annotations

import os
from typing import Dict, List

import joblib


class TfidfTextClassifier:
    """Wrapper around a scikit-learn TF-IDF pipeline."""

    def __init__(self, model_path: str = "checkpoints/tfidf_classifier.joblib") -> None:
        self.model_path = model_path
        self.pipeline = None
        self.categories: List[str] = []
        self.available = False
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        possible_paths = [self.model_path]
        backend_relative = os.path.join(os.path.dirname(__file__), self.model_path)
        project_root_relative = os.path.join(os.path.dirname(__file__), "..", self.model_path)

        for candidate in (backend_relative, project_root_relative):
            if candidate not in possible_paths:
                possible_paths.append(candidate)

        resolved_path = next((path for path in possible_paths if os.path.exists(path)), None)
        if not resolved_path:
            return

        self.model_path = resolved_path

        try:
            loaded = joblib.load(self.model_path)
            # We support either a bare pipeline or a dict containing the pipeline + metadata
            if isinstance(loaded, dict) and "pipeline" in loaded:
                self.pipeline = loaded["pipeline"]
                self.categories = loaded.get("categories") or list(getattr(self.pipeline, "classes_", []))
            else:
                self.pipeline = loaded
                self.categories = list(getattr(self.pipeline, "classes_", []))

            if not self.categories and self.pipeline is not None:
                # Fall back to looking at the classifier step explicitly
                classifier = getattr(self.pipeline, "named_steps", {}).get("clf")
                if classifier is not None and hasattr(classifier, "classes_"):
                    self.categories = list(classifier.classes_)

            if self.categories:
                self.available = True
        except Exception as exc:  # pragma: no cover - safe guard around joblib load issues
            print(f"⚠️  Failed to load TF-IDF classifier: {exc}")
            self.pipeline = None
            self.categories = []
            self.available = False

    @property
    def is_available(self) -> bool:
        return self.available and self.pipeline is not None and bool(self.categories)

    def classify(self, text: str) -> Dict:
        if not self.is_available:
            raise RuntimeError("TF-IDF classifier is not available. Train it with train_tfidf_classifier.py")

        processed_text = (text or "").strip()
        if not processed_text:
            # Mirror the structure used by the simple classifier for empty strings
            return {
                "text": "",
                "predictions": [{"label": "neutral", "score": 0.5}],
                "all_scores": {"neutral": 0.5},
                "primary_category": "neutral",
                "confidence": 0.5,
                "sentiment": "neutral",
                "emotion": "neutral",
                "model": "tfidf",
            }

        probabilities = self.pipeline.predict_proba([processed_text])[0]
        scores = {
            category: float(prob)
            for category, prob in zip(self.categories, probabilities)
        }
        predictions = [
            {"label": label, "score": float(value)}
            for label, value in scores.items()
            if value > 0.3
        ]
        predictions.sort(key=lambda item: item["score"], reverse=True)

        primary_index = int(probabilities.argmax())
        primary_category = self.categories[primary_index]
        confidence = float(probabilities[primary_index])

        sentiment = self._infer_sentiment(scores)
        emotion = self._infer_emotion(scores)

        return {
            "text": processed_text,
            "predictions": predictions,
            "all_scores": scores,
            "primary_category": primary_category,
            "confidence": confidence,
            "sentiment": sentiment,
            "emotion": emotion,
            "model": "tfidf",
        }

    @staticmethod
    def _infer_sentiment(scores: Dict[str, float]) -> str:
        if scores.get("positive", 0.0) > 0.5:
            return "positive"
        if scores.get("negative", 0.0) > 0.5 or scores.get("stress", 0.0) > 0.5:
            return "negative"
        return "neutral"

    @staticmethod
    def _infer_emotion(scores: Dict[str, float]) -> str:
        if scores.get("self_harm_high", 0.0) > 0.5 or scores.get("self_harm_low", 0.0) > 0.5:
            return "crisis"
        if scores.get("unsafe_environment", 0.0) > 0.5:
            return "unsafe"
        if scores.get("emotional_distress", 0.0) > 0.5:
            return "emotional_distress"
        if scores.get("stress", 0.0) > 0.5:
            return "stress"
        return "neutral"