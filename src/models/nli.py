"""NLI (Natural Language Inference) adapter.

Wraps a HuggingFace AutoModelForSequenceClassification + AutoTokenizer.
Different NLI checkpoints have different label orderings — we read the
model's `id2label` map at load time and produce a normalised
`{entail, neutral, contradict}` distribution.

Used in two places in the pipeline:
 1. Confidence routing: does the top retrieved chunk entail an answer to
    the question?
 2. Post-gen faithfulness: does the cited chunk entail each generated claim?
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.config import NLISpec, settings

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("HF_HOME", str(REPO_ROOT / settings.paths.hf_cache_dir))


@dataclass
class NLIResult:
    entail: float
    neutral: float
    contradict: float
    label: str  # "entailment" | "neutral" | "contradiction" | "unknown"

    @property
    def is_entailed(self) -> bool:
        return self.label == "entailment"


def _normalise_label(raw: str) -> str:
    r = raw.lower()
    if "entail" in r:
        return "entailment"
    if "contradict" in r:
        return "contradiction"
    if "neutral" in r:
        return "neutral"
    return "unknown"


class NLIModel:
    def __init__(self, spec: NLISpec, device: str | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.spec = spec
        self.id = spec.id
        device = device or settings.memory.nli_device
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(spec.hf_id)
        self.model.to(device)
        self.model.eval()
        # Map model's class indices to normalised labels
        self.idx_to_label = {
            int(k): _normalise_label(v) for k, v in self.model.config.id2label.items()
        }

    def entail(self, premise: str, hypothesis: str) -> NLIResult:
        """Run NLI for one (premise, hypothesis) pair."""
        import torch

        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

        scores = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
        for idx, p in enumerate(probs):
            label = self.idx_to_label.get(idx, "unknown")
            if label in scores:
                scores[label] += p

        # Pick the dominant label
        label = max(scores, key=scores.get) if scores else "unknown"
        return NLIResult(
            entail=scores["entailment"],
            neutral=scores["neutral"],
            contradict=scores["contradiction"],
            label=label,
        )

    def entail_batch(
        self, pairs: list[tuple[str, str]], batch_size: int = 8
    ) -> list[NLIResult]:
        """Run NLI for many (premise, hypothesis) pairs. Cheaper than calling entail() in a loop."""
        import torch

        if not pairs:
            return []
        results: list[NLIResult] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            premises = [p for p, _ in batch]
            hypotheses = [h for _, h in batch]
            inputs = self.tokenizer(
                premises,
                hypotheses,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().tolist()

            for row in probs:
                scores = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
                for idx, p in enumerate(row):
                    label = self.idx_to_label.get(idx, "unknown")
                    if label in scores:
                        scores[label] += p
                label = max(scores, key=scores.get) if scores else "unknown"
                results.append(
                    NLIResult(
                        entail=scores["entailment"],
                        neutral=scores["neutral"],
                        contradict=scores["contradiction"],
                        label=label,
                    )
                )
        return results

    def unload(self) -> None:
        try:
            import torch

            del self.model
            del self.tokenizer
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
