"""Foundation model manifold adapters and default registry."""

from prompt_manifold.adapters.base import AdapterRegistry, BaseManifoldAdapter
from prompt_manifold.adapters.dalle3 import Dalle3Adapter
from prompt_manifold.adapters.flux import FluxAdapter
from prompt_manifold.adapters.midjourney import MidjourneyAdapter
from prompt_manifold.adapters.sd3 import SD3Adapter
from prompt_manifold.adapters.sdxl import SDXLAdapter
from prompt_manifold.adapters.universal import UniversalVisionAdapter


def get_default_registry() -> AdapterRegistry:
    """Instantiate and populate default adapter registry."""
    registry = AdapterRegistry()
    registry.register(FluxAdapter())
    registry.register(SDXLAdapter())
    registry.register(SD3Adapter())
    registry.register(MidjourneyAdapter())
    registry.register(Dalle3Adapter())
    registry.register(UniversalVisionAdapter())
    return registry


__all__ = [
    "AdapterRegistry",
    "BaseManifoldAdapter",
    "Dalle3Adapter",
    "FluxAdapter",
    "MidjourneyAdapter",
    "SD3Adapter",
    "SDXLAdapter",
    "UniversalVisionAdapter",
    "get_default_registry",
]
