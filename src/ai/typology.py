"""
Destination typology matching.

Answers a question the rules cannot: what kind of place is this money going to?
A rule can see that $12,000 went somewhere new. It cannot see that the
destination reads like a remittance agent rather than a supermarket.

The answer is grounded in a committed document, data/risk_typologies.json, and
every match is reported with the typology id it came from so an investigator can
read the basis rather than take the system's word for it.

Two routes to a match, in order of preference:

  1. Embeddings. Each typology is embedded once and cached to disk; a
     destination is embedded at review time and matched by cosine similarity.
  2. Keyword overlap against the same document, used when no model is
     reachable.

Both draw on the same source text, so the fallback degrades the precision of the
match without changing what the system is allowed to claim. A destination that
resembles nothing documented is reported as unclassified rather than guessed at.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
TYPOLOGY_PATH = DATA_DIR / "risk_typologies.json"
INDEX_PATH = DATA_DIR / "typology_index.json"

UNCLASSIFIED_ID = "TYP-12"

# Below this similarity the nearest typology is not a real match, and saying so
# is more useful than forcing the destination into the closest available box.
SIMILARITY_FLOOR = 0.62


class TypologyMatcher:
    """Matches a destination against the documented typologies."""

    def __init__(self, client=None, index_path: Optional[Path] = None):
        self.client = client
        # Injectable so tests never write a cache into the repository's data
        # directory. A test once did, and a fabricated index was committed.
        self.index_path = index_path or INDEX_PATH
        self.typologies: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None
        self.vector_ids: List[str] = []
        self._load_document()

    def _load_document(self) -> None:
        try:
            document = json.loads(TYPOLOGY_PATH.read_text(encoding="utf-8"))
            self.typologies = document["typologies"]
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            log.error("Could not load typology document: %s", exc)
            self.typologies = []

    def _by_id(self, typology_id: str) -> Optional[Dict[str, Any]]:
        return next((t for t in self.typologies if t["id"] == typology_id), None)

    @staticmethod
    def _corpus_text(typology: Dict[str, Any]) -> str:
        return f"{typology['label']}. {typology['description']}"

    # ---- index construction -------------------------------------------------

    async def prepare(self) -> str:
        """
        Make the matcher ready to serve, at startup.

        Returns a short description of which route is active, for the health
        endpoint. Never raises: an unusable index means keyword matching, not a
        failed startup.
        """
        if not self.typologies:
            return "unavailable (typology document could not be read)"

        if self._load_cached_index():
            return "embeddings (cached index)"

        if self.client is None or not self.client.is_configured:
            return "keyword matching (no API key configured)"

        # The client is contracted not to raise, but startup is the one place
        # where trusting that would cost the whole application rather than one
        # request, so it is enforced here as well.
        try:
            vectors = await self.client.embed([self._corpus_text(t) for t in self.typologies])
        except Exception:
            log.exception("Embedding the typologies failed; falling back to keyword matching")
            return "keyword matching (embedding call raised)"

        if vectors is None:
            return "keyword matching (embedding call failed)"

        self.vector_ids = [t["id"] for t in self.typologies]
        self.vectors = self._normalise(np.array(vectors, dtype=np.float32))
        self._write_cached_index(vectors)
        return "embeddings (index built at startup)"

    def _load_cached_index(self) -> bool:
        if not self.index_path.exists():
            return False
        try:
            cached = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        expected = [t["id"] for t in self.typologies]
        if cached.get("typology_ids") != expected:
            log.info("Cached typology index is stale; it will be rebuilt.")
            return False

        try:
            self.vectors = self._normalise(np.array(cached["vectors"], dtype=np.float32))
        except (KeyError, ValueError):
            return False

        self.vector_ids = expected
        return True

    def _write_cached_index(self, vectors: List[List[float]]) -> None:
        """Persist the index so later startups need no network call at all."""
        try:
            self.index_path.write_text(
                json.dumps(
                    {
                        "model": self.client.embedding_model,
                        "typology_ids": self.vector_ids,
                        "vectors": vectors,
                    }
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("Could not cache the typology index: %s", exc)

    @staticmethod
    def _normalise(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    # ---- matching -----------------------------------------------------------

    async def match(self, destinations: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Match each destination to a typology.

        `destinations` are dicts with 'payee' and 'description'. The result
        carries the matched typology's id, label, posture and rationale, plus
        how the match was made, so the report can show its working.
        """
        if not destinations or not self.typologies:
            return []

        if self.vectors is not None and self.client is not None and self.client.is_configured:
            try:
                matched = await self._match_by_embedding(destinations)
            except Exception:
                log.exception("Embedding match failed; falling back to keyword matching")
                matched = None
            if matched is not None:
                return matched

        return [self._match_by_keyword(d) for d in destinations]

    async def _match_by_embedding(
        self, destinations: List[Dict[str, str]]
    ) -> Optional[List[Dict[str, Any]]]:
        queries = [f"{d['payee']}. {d.get('description', '')}".strip() for d in destinations]
        vectors = await self.client.embed(queries)
        if vectors is None:
            return None

        try:
            query_matrix = self._normalise(np.array(vectors, dtype=np.float32))
        except ValueError:
            return None

        # A cached index built by a different embedding model - or by anything
        # other than the real one - has the wrong width. Comparing against it
        # would either raise here or, worse, produce confident nonsense, so a
        # mismatch discards the index and drops to keyword matching.
        if query_matrix.shape[1] != self.vectors.shape[1]:
            log.warning(
                "Cached index has %d dimensions but the model returns %d; discarding it",
                self.vectors.shape[1], query_matrix.shape[1],
            )
            self.vectors = None
            self.vector_ids = []
            return None

        similarities = query_matrix @ self.vectors.T

        results = []
        for destination, row in zip(destinations, similarities):
            best_index = int(np.argmax(row))
            best_score = float(row[best_index])

            if best_score < SIMILARITY_FLOOR:
                typology = self._by_id(UNCLASSIFIED_ID)
                method = f"no typology within similarity floor (best {best_score:.2f})"
            else:
                typology = self._by_id(self.vector_ids[best_index])
                method = f"embedding similarity {best_score:.2f}"

            results.append(self._result(destination, typology, method))
        return results

    def _match_by_keyword(self, destination: Dict[str, str]) -> Dict[str, Any]:
        """
        Fallback matching against the document's keyword anchors.

        Anchors are scored ahead of loose prose overlap. Without that ordering a
        crypto exchange matches the remittance entry on the shared words
        "transfer" and "wire", which is the wrong answer arrived at honestly.
        """
        text = f"{destination['payee']} {destination.get('description', '')}".lower()

        best_typology, best_score = None, 0
        for typology in self.typologies:
            if typology["id"] == UNCLASSIFIED_ID:
                continue

            # A multi-word anchor is a stronger signal than a single word.
            score = sum(
                2 + anchor.count(" ")
                for anchor in typology.get("keywords", [])
                if anchor in text
            )
            if score > best_score:
                best_typology, best_score = typology, score

        if best_typology is not None:
            return self._result(
                destination, best_typology, f"keyword anchor match (score {best_score})"
            )

        # Nothing anchored. Loose prose overlap was tried here and removed: an
        # unfamiliar company name whose description reads "Outgoing Wire
        # Transfer" would match the remittance entry on the words "wire" and
        # "transfer", classifying a destination by the rail it travelled on
        # rather than by what it is. Reporting the gap is the honest answer.
        return self._result(
            destination, self._by_id(UNCLASSIFIED_ID), "no typology matched this destination"
        )

    @staticmethod
    def _result(
        destination: Dict[str, str], typology: Optional[Dict[str, Any]], method: str
    ) -> Dict[str, Any]:
        if typology is None:
            typology = {
                "id": UNCLASSIFIED_ID,
                "label": "Unclassified destination",
                "posture": "unknown",
                "why_it_matters": "The destination could not be characterised from the record.",
                "investigator_note": "Treat the category as unknown rather than inferring one.",
            }
        return {
            "payee": destination["payee"],
            "id": typology["id"],
            "label": typology["label"],
            "posture": typology["posture"],
            "why_it_matters": typology["why_it_matters"],
            "investigator_note": typology["investigator_note"],
            "matched_by": method,
            "source": "data/risk_typologies.json",
        }
