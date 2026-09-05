"""
Edge cases, malformed input, and the behaviour of the AI layer when it fails.

The load-bearing claim in this system is that the model cannot change a verdict.
Most of what follows exercises that claim by breaking the model deliberately.
"""

import json

import pytest

from src.ai.client import GeminiClient
from src.ai.narrator import validate_grounding
from src.ai.typology import TypologyMatcher
from src.models import CustomerHistory, Transaction
from src.service import InvestigationService
from tests.conftest import make_history, routine_rows, run


class TestMalformedInput:
    def test_an_empty_history_is_refused_rather_than_scored(self, analyzer):
        history = CustomerHistory(customer_id="CUST_X", account_type="Checking", transactions=[])
        with pytest.raises(ValueError):
            run(analyzer.analyze(history))

    def test_a_single_transaction_does_not_crash(self, analyzer):
        history = make_history(rows=[("2024-01-01", "Shop", 10.00, "Debit Card", "debit")])
        report = run(analyzer.analyze(history))
        assert report.risk_level == "ROUTINE"
        assert report.analysis_confidence < 0.5

    def test_the_source_field_name_is_accepted(self):
        """Histories arrive with `type`; the code reads `transaction_type`."""
        txn = Transaction(date="2024-01-01", payee="Shop", amount=10.0,
                          channel="Debit Card", type="debit")
        assert txn.transaction_type == "debit"

    def test_a_missing_description_is_tolerated(self, analyzer):
        rows = routine_rows()
        history = make_history(rows=rows)
        assert all(t.description == "" for t in history.transactions)
        run(analyzer.analyze(history))  # must not raise

    def test_identical_repeated_transactions_do_not_break_the_statistics(self, analyzer):
        """Zero variance must not produce a division by zero or a false signal."""
        rows = [(f"2024-{m:02d}-{d:02d}", "Same Payee", 100.00, "ACH", "debit")
                for m in range(1, 6) for d in (1, 8, 15, 22)]
        report = run(analyzer.analyze(make_history(rows=rows)))
        assert report.risk_level == "ROUTINE"
        assert report.risk_score == 0.0

    def test_all_credits_and_no_debits(self, analyzer):
        rows = [(f"2024-{m:02d}-{d:02d}", "Employer", 1000.00, "ACH", "credit")
                for m in range(1, 6) for d in (1, 15)]
        report = run(analyzer.analyze(make_history(rows=rows)))
        assert report.risk_level == "ROUTINE"

    def test_a_zero_amount_row_is_survivable(self, analyzer):
        rows = routine_rows() + [("2024-05-29", "Adjustment", 0.00, "ACH", "debit")]
        run(analyzer.analyze(make_history(rows=rows)))


class TestGroundingValidation:
    """A narrative that invents anything is discarded whole."""

    def test_a_fabricated_transaction_id_is_rejected(self):
        narrative = {
            "headline": "Review TXN_DEADBEEF",
            "assessment": "It moved abroad.",
            "where_to_start": [], "innocent_explanations": [], "not_established": [],
        }
        reason = validate_grounding(narrative, {"TXN_REAL0001"}, {"Known Payee"})
        assert reason is not None
        assert "TXN_DEADBEEF" in reason

    def test_a_fabricated_payee_is_rejected(self):
        narrative = {
            "headline": "Transfers to 'Offshore Shell Company Ltd'",
            "assessment": "Worth review.",
            "where_to_start": [], "innocent_explanations": [], "not_established": [],
        }
        reason = validate_grounding(narrative, {"TXN_REAL0001"}, {"Known Payee"})
        assert reason is not None
        assert "Offshore Shell Company Ltd" in reason

    def test_a_grounded_narrative_passes(self):
        narrative = {
            "headline": "Three transfers to 'Known Payee'",
            "assessment": "See TXN_REAL0001.",
            "where_to_start": ["Call the customer."],
            "innocent_explanations": ["They may have set this up themselves."],
            "not_established": ["Whether the customer authorised it."],
        }
        assert validate_grounding(narrative, {"TXN_REAL0001"}, {"Known Payee"}) is None

    def test_an_incomplete_narrative_is_rejected(self):
        narrative = {
            "headline": "", "assessment": "",
            "where_to_start": [], "innocent_explanations": [], "not_established": [],
        }
        assert validate_grounding(narrative, set(), set()) is not None


class TestAILayerFailure:
    """Whatever the model does, the findings must survive it unchanged."""

    def _service(self, analyzer, client):
        return InvestigationService(analyzer, client=client, matcher=TypologyMatcher(client))

    def _suspicious(self):
        return make_history(rows=routine_rows() + [
            ("2024-05-12", "CryptoExchange XYZ", 2500.00, "Wire", "debit"),
            ("2024-05-13", "CryptoExchange XYZ", 1500.00, "Wire", "debit"),
            ("2024-05-14", "CryptoExchange XYZ", 1000.00, "Wire", "debit"),
        ])

    def test_no_api_key_still_produces_the_full_verdict(self, analyzer):
        service = self._service(analyzer, GeminiClient(api_key=""))
        run(service.prepare())
        report = run(service.investigate(self._suspicious()))

        assert report.risk_level == "INVESTIGATE"
        assert report.findings, "findings must not depend on a model"
        assert report.investigator_narrative["available"] is False
        assert "GEMINI_API_KEY" in report.investigator_narrative["reason"]

    def test_a_raising_model_does_not_break_the_investigation(self, analyzer):
        class Exploding(GeminiClient):
            async def generate_json(self, *a, **k):
                raise RuntimeError("model exploded")

            async def embed(self, *a, **k):
                raise RuntimeError("embeddings exploded")

        client = Exploding(api_key="test-key")
        service = self._service(analyzer, client)
        run(service.prepare())
        report = run(service.investigate(self._suspicious()))

        assert report.risk_level == "INVESTIGATE"
        assert len(report.findings) >= 1
        assert report.investigator_narrative["available"] is False

    def test_an_ungrounded_model_response_is_discarded(self, analyzer):
        """A model that invents a transaction contributes nothing to the report."""
        class Fabricating(GeminiClient):
            async def generate_json(self, *a, **k):
                return {
                    "headline": "Funds moved to TXN_FAKE9999",
                    "assessment": "Invented.",
                    "where_to_start": [], "innocent_explanations": [], "not_established": [],
                }

            async def embed(self, *a, **k):
                return None

        service = self._service(analyzer, Fabricating(api_key="test-key"))
        run(service.prepare())
        report = run(service.investigate(self._suspicious()))

        assert report.investigator_narrative["available"] is False
        assert "rejected" in report.investigator_narrative["reason"]
        assert report.findings, "the rules' output is untouched by the model's failure"

    def test_enrichment_can_be_switched_off_entirely(self, analyzer):
        service = self._service(analyzer, GeminiClient(api_key=""))
        run(service.prepare())
        enriched = run(service.investigate(self._suspicious(), enrich=True))
        plain = run(service.investigate(self._suspicious(), enrich=False))

        assert plain.risk_score == enriched.risk_score
        assert plain.risk_level == enriched.risk_level
        assert [f.rule_triggered for f in plain.findings] == \
               [f.rule_triggered for f in enriched.findings]
        assert plain.investigator_narrative is None

    def test_a_clean_customer_needs_no_model_at_all(self, analyzer):
        service = self._service(analyzer, GeminiClient(api_key=""))
        run(service.prepare())
        report = run(service.investigate(make_history(rows=routine_rows())))

        assert report.risk_level == "ROUTINE"
        assert report.ai_status["enrichment_applied"] is False


class TestTypologyMatching:
    def test_an_unrecognised_destination_is_reported_as_unrecognised(self):
        """
        The system must not infer a category from the payment rail.

        Matching loosely on the description put an unknown company with the
        description "Outgoing Wire Transfer" into the remittance category,
        classifying it by how the money travelled rather than what it is.
        """
        matcher = TypologyMatcher(GeminiClient(api_key=""))
        run(matcher.prepare())
        result = run(matcher.match([
            {"payee": "Vantage Holdings Ltd", "description": "Outgoing Wire Transfer"}
        ]))
        assert result[0]["id"] == "TYP-12"
        assert result[0]["posture"] == "unknown"

    def test_recognisable_destinations_are_categorised_and_cited(self):
        matcher = TypologyMatcher(GeminiClient(api_key=""))
        run(matcher.prepare())
        results = run(matcher.match([
            {"payee": "CryptoExchange XYZ", "description": "Outgoing Wire Transfer"},
            {"payee": "Union Mortgage", "description": "Mortgage Payment"},
            {"payee": "Whole Foods", "description": "Grocery Shopping"},
        ]))
        by_payee = {r["payee"]: r for r in results}

        assert by_payee["CryptoExchange XYZ"]["id"] == "TYP-01"
        assert by_payee["Union Mortgage"]["posture"] == "benign"
        assert by_payee["Whole Foods"]["posture"] == "benign"
        for result in results:
            assert result["source"] == "data/risk_typologies.json"

    def test_matching_works_with_no_model_available(self):
        matcher = TypologyMatcher(GeminiClient(api_key=""))
        route = run(matcher.prepare())
        assert "keyword" in route


class TestTimeBudget:
    """A slow model must cost the briefing note, never the response."""

    def test_a_hanging_model_cannot_blow_the_request_budget(self, analyzer):
        import asyncio
        import time

        class Hanging(GeminiClient):
            async def generate_json(self, *a, **k):
                await asyncio.sleep(120)

            async def embed(self, *a, **k):
                await asyncio.sleep(120)

        client = Hanging(api_key="test-key")
        service = InvestigationService(analyzer, client=client, matcher=TypologyMatcher(client))
        service.ENRICHMENT_BUDGET_SECONDS = 1.0

        history = make_history(rows=routine_rows() + [
            ("2024-05-12", "CryptoExchange XYZ", 2500.00, "Wire", "debit"),
            ("2024-05-13", "CryptoExchange XYZ", 1500.00, "Wire", "debit"),
        ])

        started = time.monotonic()
        report = run(service.investigate(history))
        elapsed = time.monotonic() - started

        assert elapsed < 5, f"enrichment was not bounded: took {elapsed:.1f}s"
        assert report.risk_level == "INVESTIGATE", "the verdict must survive a hanging model"
        assert report.findings
        assert "budget" in report.investigator_narrative["reason"]


class TestApiEconomy:
    """Identical work must not be paid for twice."""

    class Counting(GeminiClient):
        """Records how many requests would actually leave the machine."""

        def __init__(self, **kw):
            super().__init__(**kw)
            self.generate_calls = 0
            self.embed_calls = 0

        async def _post(self, path, body):
            if "generateContent" in path:
                self.generate_calls += 1
                return {"candidates": [{"content": {"parts": [{"text": json.dumps({
                    "headline": "Transfers to 'CryptoExchange XYZ'",
                    "assessment": "Reviewed.",
                    "where_to_start": ["Call the customer."],
                    "innocent_explanations": ["They may have set it up."],
                    "not_established": ["Whether it was authorised."],
                })}]}}]}
            self.embed_calls += 1
            count = len(body["requests"])
            return {"embeddings": [{"values": [0.1] * 8} for _ in range(count)]}

    def _suspicious(self):
        return make_history(rows=routine_rows() + [
            ("2024-05-12", "CryptoExchange XYZ", 2500.00, "Wire", "debit"),
            ("2024-05-13", "CryptoExchange XYZ", 1500.00, "Wire", "debit"),
        ])

    def test_reviewing_the_same_customer_twice_costs_one_set_of_calls(self, analyzer):
        """
        An investigator returning to a case, or a demo clicking a customer
        repeatedly, must not re-pay for a byte-identical answer.
        """
        client = self.Counting(api_key="test-key")
        service = InvestigationService(analyzer, client=client, matcher=TypologyMatcher(client))
        run(service.prepare())

        first = run(service.investigate(self._suspicious()))
        after_first = (client.generate_calls, client.embed_calls)

        for _ in range(4):
            repeat = run(service.investigate(self._suspicious()))

        assert (client.generate_calls, client.embed_calls) == after_first, (
            "repeat reviews issued fresh API calls"
        )
        assert client.generate_calls == 1
        assert repeat.investigator_narrative == first.investigator_narrative
        assert repeat.risk_score == first.risk_score

    def test_a_payee_seen_before_is_not_embedded_again(self, analyzer):
        """The same destination recurs across customers; embed it once."""
        client = self.Counting(api_key="test-key")
        matcher = TypologyMatcher(client)
        run(matcher.prepare())

        run(matcher.match([{"payee": "CryptoExchange XYZ", "description": "Wire"}]))
        calls_after_first = client.embed_calls

        run(matcher.match([{"payee": "CryptoExchange XYZ", "description": "Wire"}]))
        assert client.embed_calls == calls_after_first

        # A destination never seen before still costs a call.
        run(matcher.match([{"payee": "Brand New Payee", "description": "Wire"}]))
        assert client.embed_calls == calls_after_first + 1

    def test_output_length_is_capped(self):
        from src.ai.client import MAX_OUTPUT_TOKENS
        assert MAX_OUTPUT_TOKENS <= 1000, "an unbounded reply is the expensive half of a call"
