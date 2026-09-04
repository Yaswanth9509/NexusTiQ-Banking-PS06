"""
Gemini client.

Speaks to the REST endpoint directly rather than through an SDK: the endpoint
is stable, the install stays small on a clean machine, and the model name is not
tied to whichever SDK release happens to know about it.

The contract this module offers the rest of the system is that it never raises.
Every call returns either a result or None, because the investigation report has
to be produced whether or not a model is reachable. A missing key, a timeout, a
rate limit and a malformed response are all the same thing to the caller: no
enrichment available this time.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

log = logging.getLogger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

DEFAULT_LLM_MODEL = "gemini-3.5-flash-lite"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

# A single request has 60 seconds end to end. The model is given a fraction of
# that so a slow call degrades to "no enrichment" rather than to a failed request.
REQUEST_TIMEOUT_SECONDS = 20.0


class GeminiClient:
    """Thin, failure-tolerant wrapper over the Gemini REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ):
        # Read at construction but tolerate absence: the app must start without a key.
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        self.llm_model = llm_model or os.getenv("GEMINI_LLM_MODEL", DEFAULT_LLM_MODEL)
        self.embedding_model = (
            embedding_model or os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        )
        self.timeout = timeout
        self._unavailable_reason: Optional[str] = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def status(self) -> Dict[str, Any]:
        """What the API layer reports about the model's availability."""
        return {
            "configured": self.is_configured,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "last_error": self._unavailable_reason,
        }

    async def generate_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> Optional[Dict[str, Any]]:
        """
        Ask the model for a JSON object matching `schema`.

        Returns the parsed object, or None if the model could not be reached or
        did not return usable JSON.
        """
        if not self.is_configured:
            self._unavailable_reason = "GEMINI_API_KEY is not set"
            return None

        body: Dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        payload = await self._post(f"{self.llm_model}:generateContent", body)
        if payload is None:
            return None

        try:
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            # A response with no candidate usually means the prompt was blocked.
            self._unavailable_reason = "model returned no usable candidate"
            log.warning("Gemini returned no candidate: %s", str(payload)[:300])
            return None

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            self._unavailable_reason = "model response was not valid JSON"
            log.warning("Gemini returned non-JSON content: %s", text[:300])
            return None

    async def embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Embed a list of texts, returning one vector each, or None on failure.

        Sent as a batch so that classifying several destinations costs one call.
        """
        if not self.is_configured:
            self._unavailable_reason = "GEMINI_API_KEY is not set"
            return None
        if not texts:
            return []

        body = {
            "requests": [
                {
                    "model": f"models/{self.embedding_model}",
                    "content": {"parts": [{"text": text}]},
                }
                for text in texts
            ]
        }

        payload = await self._post(f"{self.embedding_model}:batchEmbedContents", body)
        if payload is None:
            return None

        try:
            return [item["values"] for item in payload["embeddings"]]
        except (KeyError, TypeError):
            self._unavailable_reason = "embedding response had an unexpected shape"
            log.warning("Unexpected embedding response: %s", str(payload)[:300])
            return None

    async def _post(self, path: str, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        POST to the API, retrying once on a transient failure.

        Every failure mode collapses to None. The caller's job is to produce a
        report regardless, so there is nothing useful it could do with an exception.
        """
        url = f"{API_ROOT}/{path}"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}

        for attempt in (1, 2):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, headers=headers, json=body)

                if response.status_code == 200:
                    self._unavailable_reason = None
                    return response.json()

                # 4xx other than rate limiting will not improve on a retry.
                if response.status_code != 429 and response.status_code < 500:
                    self._unavailable_reason = (
                        f"HTTP {response.status_code} from Gemini"
                    )
                    log.warning(
                        "Gemini rejected the request (%s): %s",
                        response.status_code,
                        response.text[:300],
                    )
                    return None

                self._unavailable_reason = f"HTTP {response.status_code} from Gemini"
                log.warning("Gemini transient error %s (attempt %s)", response.status_code, attempt)

            except httpx.TimeoutException:
                self._unavailable_reason = "model call timed out"
                log.warning("Gemini call timed out (attempt %s)", attempt)
            except httpx.HTTPError as exc:
                self._unavailable_reason = f"network error: {type(exc).__name__}"
                log.warning("Gemini network error (attempt %s): %s", attempt, exc)

        return None
