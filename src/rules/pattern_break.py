"""RULE 4: Pattern Break Rule"""

from datetime import datetime
from typing import List, Optional
from src.models import Transaction, CustomerProfile, Finding
from src.rules.base import RiskRule


class PatternBreakRule(RiskRule):
    """
    Detects activity that breaks customer's established pattern.
    Triggers when activity diverges in ANY of:
      A) Payee diversity: >60% of recent 10 txns use NEW payees
      B) Channel shift: Any new channel (<20 txns) or >40% new (>=20 txns)
      C) Amount volatility: Recent std_dev > 2.0x historical std_dev
      D) Velocity: Avg time between txns drops to <1 day (spending spree)
    Weight: 0.20
    """

    def __init__(self):
        super().__init__(
            "PATTERN_BREAK",
            "Activity diverges from customer's established behavior",
            risk_weight=0.20,
        )

    def check(
        self, transactions: List[Transaction], profile: CustomerProfile
    ) -> Optional[Finding]:
        """
        Compare the customer's recent activity against their own earlier baseline.

        The two windows must not overlap. Taking the baseline as "the first 20"
        would swallow the entire history for the short records this system sees,
        leaving nothing that could ever look new, so the recent window is sized
        as a third of the history and the baseline is everything before it.
        """
        if len(transactions) < 10:
            return None  # Too little history to claim a pattern was broken

        txns = sorted(transactions, key=lambda t: t.date)

        recent_size = max(3, min(10, len(txns) // 3))
        recent_txns = txns[-recent_size:]
        baseline_txns = txns[:-recent_size]

        if len(baseline_txns) < 5:
            return None  # Baseline too thin to be a meaningful comparison

        baseline_payees = {t.payee for t in baseline_txns}
        baseline_channels = {t.channel for t in baseline_txns}
        baseline_amounts = [t.amount for t in baseline_txns]
        baseline_std = self._calculate_std(baseline_amounts)

        recent_amounts = [t.amount for t in recent_txns]
        recent_std = self._calculate_std(recent_amounts)

        # CHECK A: Payee diversity - is the customer suddenly paying strangers?
        new_payee_txns = [t for t in recent_txns if t.payee not in baseline_payees]
        if len(new_payee_txns) / len(recent_txns) > 0.60:
            new_payees = sorted({t.payee for t in new_payee_txns})
            return Finding(
                rule_triggered=self.name,
                rule_weight=self.risk_weight,
                transactions_involved=[t.transaction_id for t in new_payee_txns],
                specific_details=(
                    f"{len(new_payee_txns)} of the {len(recent_txns)} most recent transactions "
                    f"went to payees absent from the customer's earlier history: {', '.join(new_payees[:4])}"
                ),
                deviation_from_normal=(
                    f"Earlier activity ran to a settled group of payees "
                    f"({', '.join(sorted(baseline_payees)[:4])})"
                ),
                investigator_should_look=(
                    "The customer's payee list turned over almost completely. Establish whether "
                    "they opened these relationships themselves or whether the account was accessed by someone else."
                ),
                confidence=0.80,
            )

        # CHECK B: Channel shift - money leaving by unfamiliar routes
        new_channel_txns = [t for t in recent_txns if t.channel not in baseline_channels]
        channel_threshold = 0.01 if len(txns) < 20 else 0.40
        if new_channel_txns and (len(new_channel_txns) / len(recent_txns)) > channel_threshold:
            new_channels = sorted({t.channel for t in new_channel_txns})
            return Finding(
                rule_triggered=self.name,
                rule_weight=self.risk_weight,
                transactions_involved=[t.transaction_id for t in new_channel_txns],
                specific_details=(
                    f"{len(new_channel_txns)} recent transaction(s) used channels the customer "
                    f"had not used before: {', '.join(new_channels)}"
                ),
                deviation_from_normal=(
                    f"Earlier activity moved exclusively through {', '.join(sorted(baseline_channels))}"
                ),
                investigator_should_look=(
                    f"A shift to {', '.join(new_channels)} is worth confirming directly with the customer - "
                    "these routes are harder to reverse than card payments, which is why they attract misuse."
                ),
                confidence=0.75,
            )

        # CHECK C: Amount volatility - steady spending turning erratic
        if baseline_std > 0 and recent_std > (baseline_std * 2.0):
            return Finding(
                rule_triggered=self.name,
                rule_weight=self.risk_weight,
                transactions_involved=[t.transaction_id for t in recent_txns],
                specific_details=(
                    f"Recent transaction amounts vary far more than before "
                    f"(standard deviation ${recent_std:,.2f} against ${baseline_std:,.2f})"
                ),
                deviation_from_normal=(
                    f"Earlier amounts stayed close to ${sum(baseline_amounts) / len(baseline_amounts):,.2f} on average"
                ),
                investigator_should_look=(
                    "Erratic amounts can mean testing transfers before a larger one. "
                    "Check the sequence of the recent amounts for probing behaviour."
                ),
                confidence=0.65,
            )

        # CHECK D: Velocity - a sudden spending spree
        if len(txns) >= 20:
            dates = sorted(datetime.strptime(t.date, "%Y-%m-%d") for t in recent_txns)
            span = (dates[-1] - dates[0]).days
            avg_interval = span / (len(dates) - 1) if len(dates) > 1 else 0.0

            if avg_interval < 1.0:
                return Finding(
                    rule_triggered=self.name,
                    rule_weight=self.risk_weight,
                    transactions_involved=[t.transaction_id for t in recent_txns],
                    specific_details=(
                        f"{len(recent_txns)} transactions in {span} day(s) - "
                        f"roughly one every {avg_interval:.1f} days"
                    ),
                    deviation_from_normal=(
                        f"The customer's long-run rate is one transaction every "
                        f"{profile.avg_days_between_txns:.1f} days"
                    ),
                    investigator_should_look=(
                        "A sudden burst of activity can indicate someone draining an account "
                        "before access is lost. Review these transactions in date order."
                    ),
                    confidence=0.70,
                )

        return None

    @staticmethod
    def _calculate_std(values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
