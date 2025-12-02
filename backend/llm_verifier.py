"""
LLM-assisted verification stage for the multi-stage classifier.
Uses a zero-shot classification pipeline (BART-large-MNLI) to provide
an additional opinion on whether text is safe or risky. The goal is to
guard against obvious false positives/negatives without retraining.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from transformers import pipeline


@dataclass
class LLMDecision:
    """Return type for llm verification."""

    scores: Dict[str, float]
    adjustment: Optional[str] = None  # 'force_safe', 'force_risk', etc.
    rationale: Optional[str] = None


class LLMVerifier:
    """
    Wraps a zero-shot classification pipeline to act as a lightweight LLM guardrail.
    The class is instantiated once and reused by the classifier service.
    """

    def __init__(self):
        self.pipeline = None
        self.enabled = os.getenv("ENABLE_LLM_VERIFIER", "1").lower() not in ("0", "false", "no")
        self.label_map = {
            "Safe Content": "safe",
            "Neutral Conversation": "neutral",
            "Emotional Distress": "emotional_distress",
            "Self Harm Intent": "self_harm_high",
            "Mild Self Harm Intent": "self_harm_low",
            "Unsafe Environment": "unsafe_environment",
            "Stress": "stress",
        }
        self.candidate_labels = list(self.label_map.keys())
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Load the zero-shot pipeline once."""
        if not self.enabled:
            print("[LLM] verifier disabled via ENABLE_LLM_VERIFIER")
            return

        try:
            device = 0 if torch.cuda.is_available() else -1
            self.pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=device,
            )
            print("[LLM] Verifier (BART-large-MNLI) initialized")
        except Exception as exc:  # pragma: no cover - best-effort guard
            self.pipeline = None
            print(f"[LLM] Could not initialize verifier: {exc}")

    def evaluate(self, text: str) -> Optional[Dict[str, float]]:
        """Run the zero-shot classifier and normalize scores."""
        if not self.pipeline:
            return None

        result = self.pipeline(
            text,
            candidate_labels=self.candidate_labels,
            multi_label=True,
        )
        return {
            self.label_map[label]: float(score)
            for label, score in zip(result["labels"], result["scores"])
        }

    def refine(self, text: str) -> Optional[LLMDecision]:
        """
        Generate a decision that can reinforce or override baseline predictions.
        """
        scores = self.evaluate(text)
        if not scores:
            return None

        safe_score = scores.get("safe", 0.0)
        neutral_score = scores.get("neutral", 0.0)
        high_risk = max(scores.get("self_harm_high", 0.0), scores.get("self_harm_low", 0.0))
        distress = scores.get("emotional_distress", 0.0)

        # Decide whether to force safe or risk states.
        if safe_score >= 0.75 and high_risk < 0.25 and distress < 0.25:
            return LLMDecision(
                scores=scores,
                adjustment="force_safe",
                rationale="LLM safe confidence ≥ 75% with low crisis signals",
            )

        # CRITICAL: Only force_risk if BOTH conditions met:
        # 1. High LLM confidence (≥75% for self-harm, ≥80% for distress)
        # 2. Actual suicidal keywords present (checked by caller)
        # This prevents false positives from anger/toxic language
        if high_risk >= 0.75:
            # Require very high confidence for self-harm
            return LLMDecision(
                scores=scores,
                adjustment="suggest_risk",  # Changed from force_risk - caller will verify keywords
                rationale=f"LLM detected high self-harm probability ({high_risk:.1%}) - requires keyword verification",
            )
        
        if distress >= 0.80:
            # Very high threshold for distress to trigger risk
            return LLMDecision(
                scores=scores,
                adjustment="suggest_risk",  # Changed from force_risk
                rationale=f"LLM detected very high distress ({distress:.1%}) - requires keyword verification",
            )

        if neutral_score >= 0.7 and high_risk < 0.3:
            return LLMDecision(
                scores=scores,
                adjustment="reinforce_neutral",
                rationale="LLM confirmed neutral/informational intent",
            )

        return LLMDecision(scores=scores, adjustment=None, rationale=None)

