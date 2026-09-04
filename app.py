"""
Transaction Risk Investigation Assistant - TRACK_ID=PS06

Starts the whole application: API and interface together, on port 8000, from
`python app.py`.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.ai.client import GeminiClient
from src.ai.typology import TypologyMatcher
from src.analyzer import TransactionAnalyzer
from src.models import CustomerHistory, InvestigationReport
from src.rules import (
    BurstToNewPayeeRule,
    OddHoursActivityRule,
    PatternBreakRule,
    UnusuallyLargeTransferRule,
)
from src.service import InvestigationService

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("app")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sample_customers.json"
FRONTEND_DIR = ROOT / "frontend"

PORT = int(os.getenv("PORT", "8000"))


def load_customers() -> Dict[str, Dict[str, Any]]:
    """Load the sample histories, keyed by customer id."""
    try:
        records = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.error("Could not read %s: %s", DATA_PATH, exc)
        return {}
    return {record["customer_id"]: record for record in records}


CUSTOMERS = load_customers()

client = GeminiClient()
service = InvestigationService(
    analyzer=TransactionAnalyzer(
        risk_rules=[
            UnusuallyLargeTransferRule(multiplier=2.5),
            BurstToNewPayeeRule(window_days=7, min_transactions=2),
            OddHoursActivityRule(odd_hour_start=0, odd_hour_end=6, threshold=2),
            PatternBreakRule(),
        ]
    ),
    client=client,
    matcher=TypologyMatcher(client),
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Warm the typology index before serving.

    Anything that needs the network happens here, inside the startup window,
    rather than on the first request. If it fails the app still starts - the
    matcher falls back to its offline route and says so.
    """
    log.info("Loaded %d customer histories", len(CUSTOMERS))
    await service.prepare()
    log.info("Ready on port %s", PORT)
    yield


app = FastAPI(
    title="Transaction Risk Investigation Assistant",
    description="Reviews a customer's transaction history against a set of risk rules "
                "and reports whether anything needs an investigator's attention.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Liveness plus a view of what the AI layer can currently do."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "customers_loaded": len(CUSTOMERS),
        "typology_route": service.typology_route,
        "gemini": client.status,
    }


@app.get("/api/customers")
async def list_customers() -> Dict[str, Any]:
    """Every customer available to review, with enough detail to choose one."""
    summaries: List[Dict[str, Any]] = []
    for customer_id, record in CUSTOMERS.items():
        transactions = record["transactions"]
        summaries.append({
            "customer_id": customer_id,
            "account_type": record["account_type"],
            "transaction_count": len(transactions),
            "first_transaction": transactions[0]["date"] if transactions else None,
            "last_transaction": transactions[-1]["date"] if transactions else None,
        })
    summaries.sort(key=lambda item: item["customer_id"])
    return {"customers": summaries, "count": len(summaries)}


@app.get("/api/customers/{customer_id}")
async def get_customer(customer_id: str) -> Dict[str, Any]:
    """The raw history, so any figure in a report can be checked against source."""
    record = CUSTOMERS.get(customer_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return record


@app.post("/api/investigate", response_model=InvestigationReport)
async def investigate(customer_id: str, enrich: bool = True) -> InvestigationReport:
    """
    Review one customer's history.

    Set `enrich=false` for the deterministic report alone, with no model call.
    """
    record = CUSTOMERS.get(customer_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    try:
        history = CustomerHistory(**record)
    except Exception as exc:
        log.exception("Malformed history for %s", customer_id)
        raise HTTPException(
            status_code=422,
            detail=f"Stored history for {customer_id} could not be read: {exc}",
        ) from exc

    try:
        return await service.investigate(history, enrich=enrich)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Investigation failed for %s", customer_id)
        raise HTTPException(
            status_code=500, detail=f"Investigation could not be completed: {exc}"
        ) from exc


@app.post("/api/investigate/custom", response_model=InvestigationReport)
async def investigate_custom(history: CustomerHistory, enrich: bool = True) -> InvestigationReport:
    """Review a history supplied in the request body rather than a stored one."""
    try:
        return await service.investigate(history, enrich=enrich)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(str(FRONTEND_DIR / "index.html"))
else:
    @app.get("/")
    async def index() -> Dict[str, Any]:
        return {
            "service": "Transaction Risk Investigation Assistant",
            "track": "PS06",
            "endpoints": ["/health", "/api/customers", "/api/investigate", "/docs"],
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
