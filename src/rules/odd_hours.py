"""RULE 3: Odd Hours Activity Rule"""

from typing import List, Optional
from src.models import Transaction, CustomerProfile, Finding
from src.rules.base import RiskRule


class OddHoursActivityRule(RiskRule):
    """
    Detects multiple transactions during unusual hours.
    Triggers when: 2+ debit transactions between 00:00 - 05:59 AM
    Time window: Midnight to 6 AM
    Weight: 0.10
    """

    def __init__(
        self, odd_hour_start: int = 0, odd_hour_end: int = 6, threshold: int = 2
    ):
        super().__init__(
            "ODD_HOURS_ACTIVITY",
            "Multiple transactions during unusual hours (midnight-6am)",
            risk_weight=0.10,
        )
        self.odd_hour_start = odd_hour_start
        self.odd_hour_end = odd_hour_end
        self.threshold = threshold

    def check(
        self, transactions: List[Transaction], profile: CustomerProfile
    ) -> Optional[Finding]:
        """
        Check for debit activity during odd hours.

        Only transactions carrying a timestamp can be assessed. Histories with no
        time-of-day information are left unflagged rather than assumed innocent or
        guilty - the absence is reported in the analysis context instead.
        """
        timed_txns = [t for t in transactions if t.timestamp]
        if not timed_txns:
            return None

        odd_txns = []
        for t in timed_txns:
            hour = self._hour_of(t.timestamp)
            if hour is None:
                continue
            if self.odd_hour_start <= hour < self.odd_hour_end and t.transaction_type == "debit":
                odd_txns.append(t)

        if len(odd_txns) < self.threshold:
            return None

        odd_txns.sort(key=lambda t: (t.date, t.timestamp))
        listed = ", ".join(f"{t.date} {t.timestamp} (${t.amount:,.2f} to {t.payee})" for t in odd_txns[:4])
        total = sum(t.amount for t in odd_txns)

        # Confidence scales with how much of the record carries usable timestamps.
        coverage = len(timed_txns) / len(transactions)
        confidence = 0.75 if coverage >= 0.8 else 0.60

        return Finding(
            rule_triggered=self.name,
            rule_weight=self.risk_weight,
            transactions_involved=[t.transaction_id for t in odd_txns],
            specific_details=(
                f"{len(odd_txns)} debit transactions totaling ${total:,.2f} between "
                f"{self.odd_hour_start:02d}:00 and {self.odd_hour_end:02d}:00 - {listed}"
            ),
            deviation_from_normal=(
                f"The remaining {len(timed_txns) - len(odd_txns)} timestamped transactions "
                f"fall outside this window, indicating the customer normally transacts during waking hours"
            ),
            investigator_should_look=(
                "Confirm whether the customer authorised these overnight transfers. "
                "Check whether they coincide with a device or password change, or with travel "
                "across time zones, which would explain the timing innocently."
            ),
            confidence=confidence,
        )

    @staticmethod
    def _hour_of(timestamp: str) -> Optional[int]:
        """Extract the hour from an 'HH:MM' timestamp, tolerating malformed input."""
        try:
            return int(str(timestamp).split(":")[0])
        except (ValueError, IndexError, AttributeError):
            return None
