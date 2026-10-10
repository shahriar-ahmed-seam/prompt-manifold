"""Semantic drift detection and entity preservation metrics."""

import re
from typing import Set


class SemanticDriftGuard:
    """Evaluates semantic drift between sparse user queries and projected prompts."""

    STOPWORDS = {
        "a", "an", "the", "in", "on", "at", "with", "of", "by", "for", "from",
        "and", "or", "to", "is", "was", "are", "were", "it", "this", "that"
    }

    def compute_drift(self, raw_input: str, enhanced_prompt: str) -> float:
        """Calculate semantic drift penalty in [0.0, 1.0].

        0.0 = Zero semantic drift (perfect preservation of user intent).
        1.0 = Complete semantic corruption or intent hijacking.
        """
        raw_tokens = self._extract_content_tokens(raw_input)
        if not raw_tokens:
            return 0.0

        enhanced_tokens = self._extract_content_tokens(enhanced_prompt)
        if not enhanced_tokens:
            return 1.0

        # Measure recall of raw intent tokens in enhanced prompt
        preserved_tokens = raw_tokens.intersection(enhanced_tokens)
        token_recall = len(preserved_tokens) / len(raw_tokens)

        # In-distribution prompts naturally expand vocabulary, so token precision is expected
        # to be lower, but recall of key entities must remain near 1.0.
        drift_penalty = (1.0 - token_recall) * 0.85
        return round(min(1.0, max(0.0, drift_penalty)), 4)

    def verify_entity_retention(self, raw_input: str, enhanced_prompt: str) -> bool:
        """Verify whether all foundational content entities from raw prompt are retained."""
        raw_tokens = self._extract_content_tokens(raw_input)
        enhanced_tokens = self._extract_content_tokens(enhanced_prompt)
        # Content tokens with length > 3 should ideally all be present
        critical_tokens = {t for t in raw_tokens if len(t) >= 4}
        if not critical_tokens:
            return True
        missing = critical_tokens - enhanced_tokens
        return len(missing) == 0

    def _extract_content_tokens(self, text: str) -> Set[str]:
        words = re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())
        return {w for w in words if w not in self.STOPWORDS and len(w) > 1}
