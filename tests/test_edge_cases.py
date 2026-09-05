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


class TestIndexIntegrity:
    """A cached index must be trusted only when it actually fits the model."""

    def test_an_index_of_the_wrong_width_is_discarded(self, analyzer, tmp_path):
        """
        A stale or fabricated index has vectors of the wrong dimensionality.
        Using it would produce confident nonsense, so it must be dropped.
        """
        index = tmp_path / "index.json"
        matcher = TypologyMatcher(None, index_path=index)
        index.write_text(json.dumps({
            "model": "gemini-embedding-001",
            "typology_ids": [t["id"] for t in matcher.typologies],
            "vectors": [[0.1] * 8 for _ in matcher.typologies],   # far too narrow
        }), encoding="utf-8")

        class RealWidth(GeminiClient):
            async def _post(self, path, body):
                return {"embeddings": [{"values": [0.1] * 768}
                                       for _ in body["requests"]]}

        client = RealWidth(api_key="test-key")
        matcher = TypologyMatcher(client, index_path=index)
        assert run(matcher.prepare()) == "embeddings (cached index)"

        # A name no keyword anchor covers, so the embedding path is the one
        # reached and the width guard is actually exercised.
        result = run(matcher.match([{"payee": "Quorix Holdings", "description": ""}]))

        assert matcher.vectors is None, "the mismatched index should have been discarded"
        assert result[0]["id"] == "TYP-12", "an unmatchable name stays unclassified"

    def test_a_stale_index_is_rejected_when_typologies_change(self, tmp_path):
        index = tmp_path / "index.json"
        index.write_text(json.dumps({
            "model": "gemini-embedding-001",
            "typology_ids": ["TYP-01", "TYP-02"],       # far fewer than the document
            "vectors": [[0.1] * 768, [0.2] * 768],
        }), encoding="utf-8")

        matcher = TypologyMatcher(GeminiClient(api_key=""), index_path=index)
        assert "keyword" in run(matcher.prepare())


class TestMatchingPrecedence:
    """Anchors first, embeddings for the remainder - and never a guess."""

    def test_keyword_anchors_are_consulted_before_embeddings(self):
        """
        Measured against the live model the anchors were the stronger signal:
        they classified every covered destination correctly, while embeddings
        put a cryptocurrency exchange under subscriptions. Anchors therefore
        take precedence and the model only sees what they miss.
        """
        embedded = []

        class Recording(GeminiClient):
            async def _post(self, path, body):
                embedded.extend(r["content"]["parts"][0]["text"] for r in body["requests"])
                return {"embeddings": [{"values": [0.1] * 768} for _ in body["requests"]]}

        client = Recording(api_key="test-key")
        matcher = TypologyMatcher(client)
        matcher.vectors = __import__("numpy").ones((12, 768), dtype="float32")
        matcher.vector_ids = [t["id"] for t in matcher.typologies]

        results = run(matcher.match([
            {"payee": "CryptoExchange XYZ", "description": "Wire"},   # anchored
            {"payee": "Quorix Holdings", "description": ""},          # not anchored
        ]))

        assert results[0]["id"] == "TYP-01"
        assert "keyword" in results[0]["matched_by"]
        assert not any("CryptoExchange" in text for text in embedded), (
            "an anchored destination should never reach the model"
        )
        assert any("Quorix" in text for text in embedded), (
            "an unanchored destination should fall through to embeddings"
        )

    def test_a_near_tie_between_typologies_is_not_resolved_by_guessing(self):
        """
        Absolute similarity could not separate real destinations from invented
        ones - the ranges overlapped - so a winner that barely leads the
        runner-up is reported as no match rather than as the winner.
        """
        import numpy as np

        class Tied(GeminiClient):
            async def _post(self, path, body):
                return {"embeddings": [{"values": [1.0] + [0.0] * 767}
                                       for _ in body["requests"]]}

        client = Tied(api_key="test-key")
        matcher = TypologyMatcher(client)
        # Every typology sits at almost exactly the same distance.
        matcher.vectors = matcher._normalise(np.ones((12, 768), dtype="float32"))
        matcher.vector_ids = [t["id"] for t in matcher.typologies]

        result = run(matcher.match([{"payee": "Quorix Holdings", "description": ""}]))
        assert result[0]["id"] == "TYP-12"
        assert "no clear typology" in result[0]["matched_by"]


class TestInputValidation:
    """
    Malformed input is refused at the boundary with a message naming the row,
    rather than carried into the rules to fail somewhere less legible.
    """

    def test_a_negative_amount_is_refused(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Transaction(date="2024-01-01", payee="X", amount=-50.0,
                        channel="ACH", type="debit")

    def test_an_unreadable_date_is_refused_naming_the_row(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError) as excinfo:
            CustomerHistory(customer_id="C", account_type="Checking", transactions=[
                Transaction(date="2024-01-01", payee="X", amount=10.0, channel="ACH", type="debit"),
                Transaction(date="last Tuesday", payee="Y", amount=20.0, channel="ACH", type="debit"),
            ])
        message = str(excinfo.value)
        assert "transaction 1" in message, "the message should say which row is wrong"
        assert "last Tuesday" in message

    def test_a_zero_amount_is_allowed(self):
        """Adjustments and reversals legitimately post as zero."""
        txn = Transaction(date="2024-01-01", payee="Adjustment", amount=0.0,
                          channel="ACH", type="debit")
        assert txn.amount == 0.0

    def test_unusual_but_valid_payees_are_accepted(self, analyzer):
        """A name is data, not a format. Unicode and long names must pass through."""
        rows = routine_rows() + [
            ("2024-05-29", "Café Münchën 東京", 80.00, "Debit Card", "debit"),
            ("2024-05-30", "X" * 300, 80.00, "Debit Card", "debit"),
        ]
        report = run(analyzer.analyze(make_history(rows=rows)))
        assert report.risk_level in {"ROUTINE", "INVESTIGATE", "ESCALATE"}

    def test_the_api_turns_a_bad_history_into_422_not_500(self):
        """A caller sending nonsense should be told what is wrong, not see a crash."""
        from fastapi.testclient import TestClient
        import app as application

        with TestClient(application.app) as client:
            response = client.post("/api/investigate/custom", json={
                "customer_id": "C", "account_type": "Checking",
                "transactions": [{"date": "nope", "payee": "X", "amount": 10.0,
                                  "channel": "ACH", "type": "debit"}],
            })
            assert response.status_code == 422
