"""
Investigation service.

Composes the two halves of the system and keeps the seam between them explicit:

    rules  ->  risk level, score, findings        (deterministic, always runs)
    model  ->  destination context, narrative     (enrichment, may be absent)

The order is deliberate and one-way. The rules run first and their verdict is
final; the model is handed that verdict and asked to explain it. Nothing in the
AI layer can raise, lower or invent a finding, so a model that is slow, absent,
rate-limited or wrong changes how the report reads and never what it says.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from src.ai.narrator import generate_narrative
from src.ai.typology import TypologyMatcher
from src.analyzer import TransactionAnalyzer
from src.models import CustomerHistory, Finding, InvestigationReport, Transaction

log = logging.getLogger(__name__)


class InvestigationService:
    """Runs an investigation end to end."""

    # A request has 60 seconds end to end. Enrichment makes two sequential model
    # calls, each of which retries once, so its own timeouts alone could reach
    # eighty. The whole enrichment phase is therefore capped here rather than
    # relying on the individual calls to stay inside the budget between them:
    # a slow model must cost the report its briefing note, never its response.
    ENRICHMENT_BUDGET_SECONDS = 25.0

    def __init__(
        self,
        analyzer: TransactionAnalyzer,
        client=None,
        matcher: Optional[TypologyMatcher] = None,
    ):
        self.analyzer = analyzer
        self.client = client
        self.matcher = matcher
        self.typology_route = "not initialised"

    async def prepare(self) -> None:
        """Warm anything that needs warming, at startup rather than per request."""
        if self.matcher is not None:
            try:
                self.typology_route = await self.matcher.prepare()
            except Exception:
                log.exception("Typology preparation failed; matching will be unavailable")
                self.typology_route = "unavailable (preparation failed)"
            log.info("Typology matching route: %s", self.typology_route)

    async def investigate(self, history: CustomerHistory, enrich: bool = True) -> InvestigationReport:
        """
        Produce the investigation report for one customer.

        `enrich=False` returns the deterministic report alone, which is what the
        tests exercise and what a caller should use if it wants a guaranteed
        offline path.
        """
        report = await self.analyzer.analyze(history)

        if not enrich or not report.findings:
            report.ai_status = self._status(used=False, reason=(
                "enrichment not requested" if not enrich else "no findings to enrich"
            ))
            return report

        profile = self.analyzer.build_customer_profile(history)
        started = time.monotonic()

        typology_notes: List[Dict[str, Any]] = []
        narrative: Dict[str, Any] = {"available": False, "reason": "no model configured"}

        try:
            typology_notes, narrative = await asyncio.wait_for(
                self._enrich(history, profile, report.findings),
                timeout=self.ENRICHMENT_BUDGET_SECONDS,
            )
        except asyncio.TimeoutError:
            log.warning(
                "Enrichment exceeded its %.0fs budget for %s; returning the rules' verdict alone",
                self.ENRICHMENT_BUDGET_SECONDS, history.customer_id,
            )
            narrative = {
                "available": False,
                "reason": f"enrichment exceeded its {self.ENRICHMENT_BUDGET_SECONDS:.0f}s budget",
            }

        report.destination_context = typology_notes or None
        report.investigator_narrative = narrative
        report.ai_status = self._status(
            used=bool(narrative.get("available")) or bool(typology_notes),
            reason=narrative.get("reason"),
            elapsed=time.monotonic() - started,
        )
        return report

    async def _enrich(self, history, profile, findings):
        """
        The enrichment phase, run under a single overall deadline.

        Each step is independently guarded, so the narrative can still be
        produced when typology matching fails and vice versa.
        """
        typology_notes: List[Dict[str, Any]] = []
        destinations = self._destinations_under_review(history.transactions, findings)
        if self.matcher is not None and destinations:
            try:
                typology_notes = await self.matcher.match(destinations)
            except Exception:  # never let enrichment break an investigation
                log.exception("Typology matching failed; continuing without it")
                typology_notes = []

        narrative: Dict[str, Any] = {"available": False, "reason": "no model configured"}
        if self.client is not None:
            try:
                narrative = await generate_narrative(
                    client=self.client,
                    customer_id=history.customer_id,
                    profile=profile,
                    findings=findings,
                    typology_notes=typology_notes,
                )
            except Exception:
                log.exception("Narrative generation failed; continuing without it")
                narrative = {"available": False, "reason": "narrative generation raised"}

        return typology_notes, narrative

    def _destinations_under_review(
        self, transactions: List[Transaction], findings: List[Finding]
    ) -> List[Dict[str, str]]:
        """
        The distinct payees the findings actually rest on.

        Only these are looked up. Classifying every payee in the history would
        cost a far larger embedding call and tell the investigator nothing about
        the transactions in front of them.
        """
        cited_ids = {tid for finding in findings for tid in finding.transactions_involved}
        by_payee: Dict[str, str] = {}
        for txn in transactions:
            if txn.transaction_id in cited_ids and txn.payee not in by_payee:
                by_payee[txn.payee] = txn.description or ""
        return [{"payee": payee, "description": desc} for payee, desc in by_payee.items()]

    def _status(
        self, used: bool, reason: Optional[str] = None, elapsed: Optional[float] = None
    ) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "enrichment_applied": used,
            "typology_route": self.typology_route,
            "model_configured": bool(self.client and self.client.is_configured),
        }
        if elapsed is not None:
            status["enrichment_seconds"] = round(elapsed, 2)
        if self.client is not None:
            status["llm_model"] = self.client.llm_model
            status["embedding_model"] = self.client.embedding_model
        if reason and not used:
            status["reason"] = reason
        return status
