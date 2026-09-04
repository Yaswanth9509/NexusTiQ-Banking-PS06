"""
Edge cases, malformed input, and the behaviour of the AI layer when it fails.

The load-bearing claim in this system is that the model cannot change a verdict.
Most of what follows exercises that claim by breaking the model deliberately.
"""

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
