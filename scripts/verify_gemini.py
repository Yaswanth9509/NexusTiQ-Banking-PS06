"""
Verify the Gemini integration against the live API, and build the typology index.

Everything in the AI layer is covered by tests that break the model deliberately -
absent key, raising client, fabricated response, hanging call. What those cannot
confirm is that the request shapes and the model name are right, because a wrong
model name and an unreachable network fail identically from inside the
application: enrichment reports itself unavailable and the report comes back on
the rules alone.

This script closes that gap. Run it once with a key set:

    python scripts/verify_gemini.py

It checks each piece of the integration separately, says which model names the
key can actually see if one is wrong, and writes data/typology_index.json so
later startups need no network call at all.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.ai.client import API_ROOT, GeminiClient  # noqa: E402
from src.ai.typology import INDEX_PATH, TypologyMatcher  # noqa: E402
from src.analyzer import TransactionAnalyzer  # noqa: E402
from src.models import CustomerHistory  # noqa: E402
from src.rules import (  # noqa: E402
    BurstToNewPayeeRule,
    OddHoursActivityRule,
    PatternBreakRule,
    UnusuallyLargeTransferRule,
)
from src.service import InvestigationService  # noqa: E402

load_dotenv()

PASS, FAIL, WARN = "  PASS", "  FAIL", "  WARN"
results = []


def record(ok, label, detail=""):
    marker = PASS if ok else FAIL
    print(f"{marker}  {label}")
    if detail:
        for line in str(detail).splitlines():
            print(f"        {line}")
    results.append((ok, label))
    return ok


async def list_available_models(api_key):
    """Ask the key what it can see - the fastest way to diagnose a wrong name."""
    try:
        async with httpx.AsyncClient(timeout=20.0) as http:
            response = await http.get(API_ROOT, headers={"x-goog-api-key": api_key})
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text[:200]}"
        names = [
            m["name"].removeprefix("models/")
            for m in response.json().get("models", [])
        ]
        return names, None
    except httpx.HTTPError as exc:
        return None, f"{type(exc).__name__}: {exc}"


async def main():
    print("\nGemini integration check\n" + "=" * 60)

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print(f"{FAIL}  GEMINI_API_KEY is not set")
        print("\n        Set it and run again:")
        print("          export GEMINI_API_KEY=your-key      (bash)")
        print("          $env:GEMINI_API_KEY='your-key'      (PowerShell)")
        print("        or put it in a .env file at the repository root.\n")
        return 1

    client = GeminiClient()
    print(f"\n  key ............. set ({len(api_key)} chars)")
    print(f"  llm model ....... {client.llm_model}")
    print(f"  embedding model . {client.embedding_model}\n")

    # ---- what the key can see ------------------------------------------
    print("Reachability")
    names, error = await list_available_models(api_key)
    if names is None:
        record(False, "Could not list models", error)
        print("\n  The key may be invalid or the network blocked. Nothing else can pass.\n")
        return 1
    record(True, f"API reachable ({len(names)} models visible)")

    llm_ok = client.llm_model in names
    if not record(llm_ok, f"LLM model '{client.llm_model}' exists"):
        suggestions = [n for n in names if "flash" in n][:8]
        print("        Models containing 'flash' that this key can use:")
        for name in suggestions:
            print(f"          {name}")
        print("        Set GEMINI_LLM_MODEL to one of these, or update")
        print("        DEFAULT_LLM_MODEL in src/ai/client.py.")

    embed_ok = client.embedding_model in names
    if not record(embed_ok, f"Embedding model '{client.embedding_model}' exists"):
        for name in [n for n in names if "embed" in n][:8]:
            print(f"          {name}")

    # ---- structured generation -----------------------------------------
    print("\nStructured output")
    schema = {
        "type": "object",
        "properties": {"colour": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["colour", "count"],
    }
    reply = await client.generate_json(
        prompt="Reply with the colour 'blue' and the count 7.",
        schema=schema,
        system_instruction="You return JSON matching the schema exactly.",
    )
    if record(reply is not None, "generateContent returned parsed JSON",
              client.status.get("last_error") if reply is None else ""):
        record(isinstance(reply, dict) and "colour" in reply,
               "Response honoured responseSchema", json.dumps(reply))

    # ---- embeddings ------------------------------------------------------
    print("\nEmbeddings")
    vectors = await client.embed(["a cryptocurrency exchange", "a supermarket"])
    if record(vectors is not None, "batchEmbedContents returned vectors",
              client.status.get("last_error") if vectors is None else ""):
        record(len(vectors) == 2, f"Returned one vector per input ({len(vectors)})")
        record(len(vectors[0]) > 100, f"Vector dimensionality is {len(vectors[0])}")

    # ---- caching ---------------------------------------------------------
    print("\nRequest economy")
    before = client.status["calls_made"]
    await client.embed(["a cryptocurrency exchange", "a supermarket"])
    record(client.status["calls_made"] == before,
           "Repeat embeddings served from cache, no second call")

    # ---- build and persist the index --------------------------------------
    print("\nTypology index")
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
    matcher = TypologyMatcher(client)
    route = await matcher.prepare()
    if record("embeddings" in route, f"Index built via embeddings ({route})"):
        record(INDEX_PATH.exists(), f"Written to {INDEX_PATH.relative_to(ROOT)}")

        matched = await matcher.match([
            {"payee": "CryptoExchange XYZ", "description": "Outgoing Wire Transfer"},
            {"payee": "Whole Foods", "description": "Grocery Shopping"},
            {"payee": "Union Mortgage", "description": "Mortgage Payment"},
        ])
        for result in matched:
            print(f"        {result['payee']:<22} -> {result['id']} "
                  f"{result['posture']:<9} ({result['matched_by']})")
        crypto = next(r for r in matched if r["payee"] == "CryptoExchange XYZ")
        record(crypto["posture"] == "elevated",
               "Cryptocurrency exchange categorised as elevated risk")

    # ---- end to end -------------------------------------------------------
    print("\nEnd to end")
    records = {c["customer_id"]: c
               for c in json.loads((ROOT / "data" / "sample_customers.json").read_text("utf-8"))}
    service = InvestigationService(
        analyzer=TransactionAnalyzer(risk_rules=[
            UnusuallyLargeTransferRule(2.5), BurstToNewPayeeRule(7, 2),
            OddHoursActivityRule(), PatternBreakRule(),
        ]),
        client=client, matcher=matcher,
    )
    await service.prepare()

    report = await service.investigate(CustomerHistory(**records["CUST_009"]))
    record(report.risk_level == "ESCALATE", f"CUST_009 -> {report.risk_level} {report.risk_score}")

    narrative = report.investigator_narrative or {}
    if record(narrative.get("available") is True, "Narrative generated and passed grounding",
              narrative.get("reason", "")):
        print()
        print(f"        headline: {narrative['headline']}")
        print(f"        {narrative['assessment']}")
        for item in narrative.get("where_to_start", [])[:3]:
            print(f"          start: {item}")
        for item in narrative.get("innocent_explanations", [])[:2]:
            print(f"          innocent: {item}")
        print()
        record(report.ai_status.get("enrichment_seconds", 99) < 25,
               f"Completed in {report.ai_status.get('enrichment_seconds')}s (budget 25s)")

    # ---- summary ----------------------------------------------------------
    passed = sum(1 for ok, _ in results if ok)
    print("\n" + "=" * 60)
    print(f"{passed}/{len(results)} checks passed")

    failures = [label for ok, label in results if not ok]
    if failures:
        print("\nFailed:")
        for label in failures:
            print(f"  - {label}")
        print("\nThe application still runs with any of these failing: enrichment")
        print("reports itself unavailable and reports come back on the rules alone.\n")
        return 1

    print("\nAll good. Commit data/typology_index.json so judges start without")
    print("a network call:\n")
    print("  git add data/typology_index.json && git commit -m 'Precomputed typology index'\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
