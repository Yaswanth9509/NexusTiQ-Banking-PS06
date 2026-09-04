"""
Each rule, tested against what it should catch and what it must leave alone.

The second half matters as much as the first. A rule that fires on everything
is not a strict rule, it is a broken one, so most of these cases are ordinary
behaviour that must come back clean.
"""

from src.rules import (
    BurstToNewPayeeRule,
    OddHoursActivityRule,
    PatternBreakRule,
    UnusuallyLargeTransferRule,
)
from tests.conftest import make_history, routine_rows


def profile_for(analyzer, history):
    return analyzer.build_customer_profile(history)


class TestUnusuallyLargeTransfer:
    def test_flags_a_large_payment_to_an_unknown_destination(self, analyzer):
        history = make_history(rows=routine_rows() + [
            ("2024-05-20", "Meridian Overseas Bank", 12000.00, "Wire", "debit"),
        ])
        finding = UnusuallyLargeTransferRule().check(
            history.transactions, profile_for(analyzer, history)
        )

        assert finding is not None
        assert "12,000" in finding.specific_details
        assert "Meridian Overseas Bank" in finding.specific_details
        assert len(finding.transactions_involved) == 1

    def test_leaves_the_monthly_rent_alone(self, analyzer):
        """
        Rent dwarfs the median every month and is not a signal.

        Comparing against the median alone flagged rent or a mortgage on
        sixteen of the twenty sample customers.
        """
        history = make_history(rows=routine_rows())
        finding = UnusuallyLargeTransferRule().check(
            history.transactions, profile_for(analyzer, history)
        )
        assert finding is None

    def test_leaves_a_standing_investment_sweep_alone(self, analyzer):
        """A recurring transfer is a plan, however large it is."""
        rows = routine_rows()
        for month in range(1, 6):
            rows.append((f"2024-{month:02d}-20", "Fairmont Brokerage", 5000.00, "Wire", "debit"))

        history = make_history(rows=rows)
        finding = UnusuallyLargeTransferRule().check(
            history.transactions, profile_for(analyzer, history)
        )
        assert finding is None, "a monthly sweep to the same brokerage is not an anomaly"

    def test_flags_a_cluster_to_one_stranger_despite_the_repetition(self, analyzer):
        """
        Three transfers in three days repeat, but they do not make a relationship.

        Exempting any payee seen twice would let a burst immunise itself.
        """
        rows = routine_rows() + [
            ("2024-05-14", "Vantage Holdings", 3500.00, "Wire", "debit"),
            ("2024-05-15", "Vantage Holdings", 2800.00, "Wire", "debit"),
            ("2024-05-16", "Vantage Holdings", 2200.00, "Wire", "debit"),
        ]
        history = make_history(rows=rows)
        finding = UnusuallyLargeTransferRule().check(
            history.transactions, profile_for(analyzer, history)
        )

        assert finding is not None
        assert "no standing arrangement" in finding.deviation_from_normal

    def test_ignores_money_coming_in(self, analyzer):
        rows = routine_rows() + [("2024-05-20", "Inheritance", 90000.00, "Wire", "credit")]
        history = make_history(rows=rows)
        finding = UnusuallyLargeTransferRule().check(
            history.transactions, profile_for(analyzer, history)
        )
        assert finding is None, "a large credit is not a payment out"

    def test_small_amounts_never_qualify(self, analyzer):
        """A customer whose median is tiny still should not be flagged on a shop."""
        rows = [(f"2024-{m:02d}-{d:02d}", "Coffee", 4.00, "Debit Card", "debit")
                for m in range(1, 6) for d in (3, 10, 17, 24)]
        rows.append(("2024-05-26", "Bookshop", 40.00, "Debit Card", "debit"))

        history = make_history(rows=rows)
        finding = UnusuallyLargeTransferRule().check(
            history.transactions, profile_for(analyzer, history)
        )
        assert finding is None, "$40 is 10x this median and still not worth an investigator"


class TestBurstToNewPayee:
    def test_flags_repeated_transfers_to_a_newly_added_payee(self, analyzer):
        rows = routine_rows() + [
            ("2024-05-12", "CryptoExchange XYZ", 2500.00, "Wire", "debit"),
            ("2024-05-13", "CryptoExchange XYZ", 1500.00, "Wire", "debit"),
            ("2024-05-14", "CryptoExchange XYZ", 1000.00, "Wire", "debit"),
        ]
        history = make_history(rows=rows)
        finding = BurstToNewPayeeRule().check(
            history.transactions, profile_for(analyzer, history)
        )

        assert finding is not None
        assert len(finding.transactions_involved) == 3
        assert "5,000" in finding.specific_details

    def test_a_single_payment_to_a_new_payee_is_not_a_burst(self, analyzer):
        rows = routine_rows() + [("2024-05-12", "New Shop", 2500.00, "Wire", "debit")]
        history = make_history(rows=rows)
        assert BurstToNewPayeeRule().check(
            history.transactions, profile_for(analyzer, history)
        ) is None

    def test_payments_spread_beyond_the_window_are_not_a_burst(self, analyzer):
        rows = routine_rows() + [
            ("2024-05-01", "Occasional Payee", 900.00, "Wire", "debit"),
            ("2024-05-20", "Occasional Payee", 900.00, "Wire", "debit"),
        ]
        history = make_history(rows=rows)
        assert BurstToNewPayeeRule(window_days=7).check(
            history.transactions, profile_for(analyzer, history)
        ) is None, "19 days apart is not a burst"

    def test_long_standing_payees_are_never_treated_as_new(self, analyzer):
        """The weekly supermarket run must not read as a burst."""
        history = make_history(rows=routine_rows())
        assert BurstToNewPayeeRule().check(
            history.transactions, profile_for(analyzer, history)
        ) is None


class TestOddHours:
    def test_flags_repeated_overnight_payments(self, analyzer):
        rows = [(d, p, a, c, k, "14:30") for d, p, a, c, k in routine_rows()]
        rows += [
            ("2024-05-14", "QuickPay", 420.00, "Online", "debit", "03:14"),
            ("2024-05-15", "QuickPay", 380.00, "Online", "debit", "02:47"),
        ]
        history = make_history(rows=rows)
        finding = OddHoursActivityRule().check(
            history.transactions, profile_for(analyzer, history)
        )

        assert finding is not None
        assert len(finding.transactions_involved) == 2

    def test_one_late_payment_is_not_a_pattern(self, analyzer):
        rows = [(d, p, a, c, k, "14:30") for d, p, a, c, k in routine_rows()]
        rows.append(("2024-05-14", "QuickPay", 420.00, "Online", "debit", "03:14"))
        history = make_history(rows=rows)
        assert OddHoursActivityRule().check(
            history.transactions, profile_for(analyzer, history)
        ) is None

    def test_a_history_without_times_is_not_judged(self, analyzer):
        """
        Absence of timestamps is a gap in the record, not evidence either way.

        The specification's own draft read a `timestamp` attribute that no
        transaction carried, so the rule could never fire at all.
        """
        history = make_history(rows=routine_rows())
        assert OddHoursActivityRule().check(
            history.transactions, profile_for(analyzer, history)
        ) is None

    def test_overnight_credits_are_not_flagged(self, analyzer):
        rows = [(d, p, a, c, k, "14:30") for d, p, a, c, k in routine_rows()]
        rows += [
            ("2024-05-14", "Employer Ltd", 3500.00, "ACH", "credit", "03:14"),
            ("2024-05-15", "Employer Ltd", 3500.00, "ACH", "credit", "02:47"),
        ]
        history = make_history(rows=rows)
        assert OddHoursActivityRule().check(
            history.transactions, profile_for(analyzer, history)
        ) is None, "a batch salary run landing overnight is not customer activity"

    def test_malformed_timestamps_do_not_crash_the_rule(self, analyzer):
        rows = [(d, p, a, c, k, "not-a-time") for d, p, a, c, k in routine_rows()]
        history = make_history(rows=rows)
        assert OddHoursActivityRule().check(
            history.transactions, profile_for(analyzer, history)
        ) is None


class TestPatternBreak:
    def test_flags_a_wholesale_change_of_payees(self, analyzer):
        rows = routine_rows()
        for index, payee in enumerate((
            "Prepaid Reload", "Instant Remit", "Voucher Direct", "TopUp Wallet",
            "Global Send", "CashPoint", "Rapid Transfer", "Voucher Hub", "Swift Remit",
        )):
            rows.append((f"2024-06-{index + 1:02d}", payee, 260.00, "Wire", "debit"))

        history = make_history(rows=rows)
        finding = PatternBreakRule().check(
            history.transactions, profile_for(analyzer, history)
        )

        assert finding is not None
        assert finding.transactions_involved

    def test_a_settled_routine_does_not_break_its_own_pattern(self, analyzer):
        history = make_history(rows=routine_rows())
        assert PatternBreakRule().check(
            history.transactions, profile_for(analyzer, history)
        ) is None

    def test_a_short_history_is_not_compared_against_itself(self, analyzer):
        """
        With too little behind the recent window there is no baseline to break.

        Taking the first twenty transactions as the baseline, as originally
        specified, meant the baseline swallowed the whole history for records
        this size and nothing could ever look new.
        """
        rows = [(f"2024-01-{d:02d}", "Shop", 50.00, "Debit Card", "debit") for d in range(1, 12)]
        history = make_history(rows=rows)
        assert PatternBreakRule().check(
            history.transactions, profile_for(analyzer, history)
        ) is None
