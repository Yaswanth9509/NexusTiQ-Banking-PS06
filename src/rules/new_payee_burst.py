"""RULE 2: Burst to New Payee Rule"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Optional
from src.models import Transaction, CustomerProfile, Finding
from src.rules.base import RiskRule


class BurstToNewPayeeRule(RiskRule):
    """
    Detects multiple rapid transactions to a newly added payee.
    Triggers when: 2+ transactions to SAME new payee within 7 days
    Definition: New payee = not in established_payees (appeared <2 times)
    Weight: 0.40 (HIGHEST WEIGHT)
    """

    def __init__(self, window_days: int = 7, min_transactions: int = 2):
        super().__init__(
            "BURST_TO_NEW_PAYEE",
            "Multiple rapid transactions to a newly added payee",
            risk_weight=0.40,
        )
        self.window_days = window_days
        self.min_transactions = min_transactions

    def check(
        self, transactions: List[Transaction], profile: CustomerProfile
    ) -> Optional[Finding]:
        """
        Check for bursts to newly added payees.

        A payee is "newly added" if it first appears only after the customer's
        baseline is already established - that is, it is absent from the earlier
        half of the observed history. Counting appearances across the whole
        history cannot work here: the burst itself would make its own payee look
        established and the rule could never fire.
        """
        if len(transactions) < self.min_transactions:
            return None

        dated = [(self._parse(t.date), t) for t in transactions]
        dated.sort(key=lambda pair: pair[0])

        first_date = dated[0][0]
        last_date = dated[-1][0]
        span_days = (last_date - first_date).days

        if span_days <= 0:
            return None

        # Payees seen in the first half of the history form the established set.
        baseline_cutoff = first_date + timedelta(days=span_days // 2)
        established_payees = {t.payee for d, t in dated if d <= baseline_cutoff}

        # Group the remaining payees' transactions by payee.
        new_payee_txns = defaultdict(list)
        for d, t in dated:
            if t.payee not in established_payees:
                new_payee_txns[t.payee].append((d, t))

        # Report the most significant burst (largest total amount moved).
        bursts = []
        for payee, entries in new_payee_txns.items():
            if len(entries) < self.min_transactions:
                continue

            window = self._tightest_window(entries)
            if window is None:
                continue

            burst_entries, days_span = window
            total_amount = sum(t.amount for _, t in burst_entries)
            bursts.append((total_amount, payee, burst_entries, days_span))

        if not bursts:
            return None

        total_amount, payee, burst_entries, days_span = max(bursts, key=lambda b: b[0])
        burst_txns = [t for _, t in burst_entries]
        first_txn = burst_txns[0]

        return Finding(
            rule_triggered=self.name,
            rule_weight=self.risk_weight,
            transactions_involved=[t.transaction_id for t in burst_txns],
            specific_details=(
                f"{len(burst_txns)} transactions totaling ${total_amount:,.2f} to new payee "
                f"'{payee}' within {days_span} day(s) ({first_txn.date} to {burst_txns[-1].date})"
            ),
            deviation_from_normal=(
                f"'{payee}' does not appear in the customer's established payees "
                f"({', '.join(sorted(established_payees)[:4])})"
            ),
            investigator_should_look=(
                f"First transfer to '{payee}' was {first_txn.date} for ${first_txn.amount:,.2f} "
                f"via {first_txn.channel}. Confirm with the customer that this payee was added "
                f"by them and that the transfers were authorised."
            ),
            confidence=0.95,
        )

    def _tightest_window(self, entries):
        """
        Return the largest cluster of transactions falling inside window_days,
        together with the span it covers, or None if no such cluster exists.
        """
        best = None
        for i in range(len(entries)):
            cluster = [
                entries[j]
                for j in range(i, len(entries))
                if (entries[j][0] - entries[i][0]).days <= self.window_days
            ]
            if len(cluster) < self.min_transactions:
                continue
            span = (cluster[-1][0] - cluster[0][0]).days
            if best is None or len(cluster) > len(best[0]):
                best = (cluster, span)
        return best

    @staticmethod
    def _parse(date_str: str) -> datetime:
        return datetime.strptime(date_str, "%Y-%m-%d")
