"""AI enrichment layer - kept separate from the deterministic rule engine."""

from src.ai.client import GeminiClient
from src.ai.narrator import generate_narrative
from src.ai.typology import TypologyMatcher

__all__ = ["GeminiClient", "generate_narrative", "TypologyMatcher"]
