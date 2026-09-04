"""Profile construction, maturity, scoring, and the grounding guarantee."""

import pytest


from tests.conftest import make_history, routine_rows, run


class TestProfile:
    def test_debit_median_ignores_credits(self, analyzer):
        """
        The debit median must not be dragged up by salary.

        This is the single assumption the large-transfer rule rests on: a median
        computed over everything sits between the spending and the salary and
        describes neither.
        """
        history = make_history(rows=routine_rows())
        profile = analyzer.build_customer_profile(history)

        assert profile.median_debit_amount < 200, "salary credits leaked into the debit median"
        assert profile.median_amount >= profile.median_debit_amount

    def test_established_payees_need_two_appearances(self, analyzer):
        history = make_history(rows=routine_rows() + [
            ("2024-05-28", "One Off Shop", 40.00, "Debit Card", "debit"),
        ])
        profile = analyzer.build_customer_profile(history)

        assert "Supermarket" in profile.typical_payees
        assert "One Off Shop" not in profile.typical_payees

    def test_observation_window_survives_unsorted_input(self, analyzer):
        """A history that arrives out of order must still yield the right window."""
        rows = routine_rows()
        shuffled = rows[len(rows) // 2:] + rows[: len(rows) // 2]

        ordered = analyzer.build_customer_profile(make_history(rows=rows))
        jumbled = analyzer.build_customer_profile(make_history(rows=shuffled))

        assert ordered.observation_days == jumbled.observation_days
        assert ordered.observation_days > 100


class TestMaturity:
    def test_too_few_transactions_is_insufficient(self, analyzer):
        history = make_history(rows=[
            ("2024-01-01", "Employer Ltd", 3000.00, "ACH", "credit"),
            ("2024-02-01", "Employer Ltd", 3000.00, "ACH", "credit"),
            ("2024-03-01", "Employer Ltd", 3000.00, "ACH", "credit"),
        ])
        profile = analyzer.build_customer_profile(history)
        assert profile.maturity_level == "INSUFFICIENT"

    def test_short_window_is_emerging(self, analyzer):
        rows = [(f"2024-01-{day:02d}", "Shop", 50.00, "Debit Card", "debit") for day in range(1, 16)]
        profile = analyzer.build_customer_profile(make_history(rows=rows))
        assert profile.maturity_level == "EMERGING"

    def test_a_month_of_history_is_established(self, analyzer):
        profile = analyzer.build_customer_profile(make_history(rows=routine_rows()))
        assert profile.maturity_level == "ESTABLISHED"
        assert profile.is_mature is True

    def test_insufficient_history_reports_the_gap_rather_than_a_verdict(self, analyzer):
        history = make_history(rows=[
            ("2024-01-01", "Employer Ltd", 3000.00, "ACH", "credit"),
            ("2024-01-15", "Shop", 50.00, "Debit Card", "debit"),
        ])
        report = run(analyzer.analyze(history))

        assert report.risk_level == "ROUTINE"
        assert report.findings is None
        assert report.analysis_confidence < 0.5, "a guess from two rows must not look confident"
        assert "insufficient" in report.summary.lower()


class TestRiskScoring:
    def test_no_findings_scores_zero(self, analyzer):
        assert analyzer._calculate_risk_score([]) == 0.0

    def test_bands_do_not_overlap_and_rise_with_count(self, analyzer, histories):
        """
        One finding must never outrank two, and two must never outrank three.

        The published formula multiplied raw weights of 0.10 to 0.40 by 0.30,
        which pinned every band to its floor and made a single burst finding
        indistinguishable from a single odd-hours finding.
        """
        from src.models import Finding

        def finding(weight):
            return Finding(
                rule_triggered="X", rule_weight=weight, transactions_involved=["TXN_1"],
                specific_details="d", deviation_from_normal="d",
                investigator_should_look="d", confidence=0.9,
            )

        one_light = analyzer._calculate_risk_score([finding(0.10)])
        one_heavy = analyzer._calculate_risk_score([finding(0.40)])
        two = analyzer._calculate_risk_score([finding(0.40), finding(0.30)])
        three = analyzer._calculate_risk_score([finding(0.40), finding(0.30), finding(0.10)])

        assert one_light < one_heavy, "rule weight must actually separate single findings"
        assert 0.30 <= one_light and one_heavy <= 0.60
        assert one_heavy < two <= 0.75
        assert two < three <= 1.0

    def test_three_findings_reach_the_escalation_threshold(self, analyzer):
        from src.models import Finding

        findings = [
            Finding(rule_triggered=f"R{i}", rule_weight=w, transactions_involved=["TXN_1"],
                    specific_details="d", deviation_from_normal="d",
                    investigator_should_look="d", confidence=0.9)
            for i, w in enumerate((0.40, 0.30, 0.20))
        ]
        assert analyzer._calculate_risk_score(findings) >= 0.80


class TestGrounding:
    """The system must never cite a transaction that is not in the input."""

    def test_every_cited_transaction_exists_in_the_source_history(
        self, analyzer, histories, sample_records
    ):
        checked = 0
        for customer_id, history in histories.items():
            report = run(analyzer.analyze(history))
            source_ids = {t.transaction_id for t in history.transactions}

            for finding in report.findings or []:
                assert finding.transactions_involved, (
                    f"{customer_id}/{finding.rule_triggered} cited no transactions"
                )
                for txn_id in finding.transactions_involved:
                    assert txn_id in source_ids, (
                        f"{customer_id}/{finding.rule_triggered} cited {txn_id}, "
                        f"which is not in the history"
                    )
                    checked += 1
        assert checked > 0, "no citations were exercised"

    def test_every_finding_carries_its_explanation(self, analyzer, histories):
        for customer_id, history in histories.items():
            report = run(analyzer.analyze(history))
            for finding in report.findings or []:
                assert finding.deviation_from_normal.strip(), f"{customer_id}: no baseline comparison"
                assert finding.investigator_should_look.strip(), f"{customer_id}: no guidance"
                assert 0.0 < finding.confidence <= 1.0

    def test_transaction_ids_are_stable_across_parses(self, sample_records):
        """
        A report is only traceable if the same row keeps the same reference.

        Identifiers were originally random per parse, so two calls produced
        different ids for one transaction and nothing could be followed back.
        """
        from src.models import CustomerHistory

        record = sample_records["CUST_003"]
        first = [t.transaction_id for t in CustomerHistory(**record).transactions]
        second = [t.transaction_id for t in CustomerHistory(**record).transactions]
        assert first == second

    def test_ids_are_present_in_the_committed_data(self, sample_records):
        for customer_id, record in sample_records.items():
            for txn in record["transactions"]:
                assert txn.get("transaction_id"), f"{customer_id} has a row with no id in the data file"

    def test_ids_derived_when_absent_match_the_committed_ones(self, sample_records):
        """A history supplied without ids must resolve to the same references."""
        from src.models import CustomerHistory

        record = sample_records["CUST_009"]
        stripped = {
            **record,
            "transactions": [
                {k: v for k, v in t.items() if k != "transaction_id"}
                for t in record["transactions"]
            ],
        }
        assert [t.transaction_id for t in CustomerHistory(**stripped).transactions] == \
               [t["transaction_id"] for t in record["transactions"]]
