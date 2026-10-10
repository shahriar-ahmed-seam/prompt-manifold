"""Base adapter protocol and registry for target foundation model manifolds."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from prompt_manifold.schemas import ExpansionConfig, IntentSlots, ManifoldTarget


class BaseManifoldAdapter(ABC):
    """Abstract base class for foundation model prompt manifold adapters."""

    @property
    @abstractmethod
    def target(self) -> ManifoldTarget:
        """Target foundation model identifier."""
        pass

    @abstractmethod
    def format_prompt(
        self, slots: IntentSlots, config: ExpansionConfig
    ) -> Tuple[str, Dict[str, Any]]:
        """Project semantic slots into target-specific syntax and token structure."""
        pass

    @abstractmethod
    def synthesize_negative(
        self, slots: IntentSlots, config: ExpansionConfig
    ) -> str:
        """Synthesize model-specific negative steering vector."""
        pass


class AdapterRegistry:
    """Registry managing available manifold adapters."""

    def __init__(self):
        self._adapters: Dict[ManifoldTarget, BaseManifoldAdapter] = {}

    def register(self, adapter: BaseManifoldAdapter) -> None:
        """Register an adapter instance."""
        self._adapters[adapter.target] = adapter

    def get(self, target: ManifoldTarget) -> BaseManifoldAdapter:
        """Retrieve adapter for target manifold."""
        if target not in self._adapters:
            # Fallback to Universal Vision adapter if specific target not registered
            if ManifoldTarget.UNIVERSAL_VISION in self._adapters:
                return self._adapters[ManifoldTarget.UNIVERSAL_VISION]
            raise KeyError(f"No adapter registered for target '{target}'")
        return self._adapters[target]
