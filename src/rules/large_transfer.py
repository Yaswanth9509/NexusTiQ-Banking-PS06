"""RULE 1: Unusually Large Transfer Rule"""

from datetime import datetime
from typing import List, Optional
from src.models import Transaction, CustomerProfile, Finding
from src.rules.base import RiskRule


class UnusuallyLargeTransferRule(RiskRule):
    """
    Detects transactions significantly larger than customer's typical amount.
    Threshold: >2.5x median (or >3.5x for EMERGING profiles)
    Only applies to debit transactions
    Weight: 0.30
    """

    def __init__(self, multiplier: float = 2.5):
        super().__init__(
            "UNUSUALLY_LARGE_TRANSFER",
            "Transaction amount significantly larger than customer's typical",
            risk_weight=0.30,
        )
        self.multiplier = multiplier

    # A payment may exceed the threshold and still be routine if the customer
    # sends comparable amounts to that same payee regularly. This is the margin
    # allowed above the largest previous payment to an established payee.
    ESTABLISHED_PAYEE_TOLERANCE = 1.3

    # A standing arrangement recurs over time rather than all at once; payments
    # clustered inside three weeks do not establish one. In a history too short
    # for that to be possible the requirement scales down with the window, so a
    # young account is not held to a standard its data cannot meet.
    RELATIONSHIP_MIN_DAYS = 21
    RELATIONSHIP_WINDOW_FRACTION = 0.25

    # Below this, a payment is not worth an investigator's time whatever the
    # arithmetic says. A customer whose typical spend is $50 will clear a 2.5x
    # threshold on an ordinary shop, and a desk that surfaces those stops
    # reading the ones that matter.
    MATERIALITY_FLOOR = 500.00

    def check(
        self, transactions: List[Transaction], profile: CustomerProfile
    ) -> Optional[Finding]:
        """
        Flag a payment out that is large both for this customer and for its
        destination.

        Size alone is not a signal. Rent, a mortgage and a standing investment
        sweep all dwarf a customer's median payment every single month, and
        flagging them would bury the investigator in noise. What matters is a
        large amount going somewhere the customer does not routinely send
        large amounts.
        """
        debit_txns = [t for t in transactions if t.transaction_type == "debit"]
        if not debit_txns:
            return None

        # A forming profile earns a looser threshold - there is less evidence
        # behind its idea of "normal", so it should accuse less readily.
        threshold_multiplier = 3.5 if profile.maturity_level == "EMERGING" else self.multiplier
        threshold = profile.median_debit_amount * threshold_multiplier

        candidates = [
            t for t in debit_txns
            if t.amount > threshold and t.amount >= self.MATERIALITY_FLOOR
        ]
        if not candidates:
            return None

        flagged = [
            t for t in candidates
            if not self._is_routine_for_payee(t, debit_txns, profile.observation_days)
        ]
        if not flagged:
            return None

        largest = max(flagged, key=lambda t: t.amount)
        multiple = largest.amount / profile.median_debit_amount if profile.median_debit_amount else 0
        payee_note, guidance = self._describe_payee_history(largest, debit_txns)

        return Finding(
            rule_triggered=self.name,
            rule_weight=self.risk_weight,
            transactions_involved=[largest.transaction_id],
            specific_details=(
                f"Debit of ${largest.amount:,.2f} on {largest.date} to '{largest.payee}' "
                f"via {largest.channel}"
            ),
            deviation_from_normal=(
                f"That is {multiple:.1f}x the customer's median payment of "
                f"${profile.median_debit_amount:,.2f} (threshold ${threshold:,.2f}), and "
                f"{payee_note}"
            ),
            investigator_should_look=guidance,
            confidence=0.90 if profile.maturity_level == "ESTABLISHED" else 0.70,
        )

    def _describe_payee_history(self, txn: Transaction, debit_txns: List[Transaction]):
        """
        Describe what the customer has previously sent this payee, and say what
        that means for the investigator.

        The distinction that matters is between a payee with a genuine past and
        one whose only other payments arrived alongside this one. Reporting the
        second as "2 previous payments" would imply a history that does not
        exist.
        """
        siblings = [t for t in debit_txns if t.payee == txn.payee and t is not txn]

        if not siblings:
            return (
                f"'{txn.payee}' has received no other payment in this history",
                f"Establish what the ${txn.amount:,.2f} to '{txn.payee}' was for and whether "
                f"the customer set it up themselves. A payment this size to a destination that "
                f"appears once and never again is the kind worth a call.",
            )

        dates = sorted(datetime.strptime(t.date, "%Y-%m-%d") for t in siblings + [txn])
        span = (dates[-1] - dates[0]).days
        prior_max = max(t.amount for t in siblings)

        if span < self.RELATIONSHIP_MIN_DAYS:
            return (
                f"the only other payments to '{txn.payee}' were {len(siblings)} more in the same "
                f"{span}-day period (largest ${prior_max:,.2f}), so there is no standing "
                f"arrangement behind them",
                f"Treat the ${txn.amount:,.2f} as one leg of a short sequence to '{txn.payee}' "
                f"rather than in isolation. Confirm with the customer that they set this payee up "
                f"and intended the full amount that left.",
            )

        return (
            f"previous payments to '{txn.payee}' span {span} days and peaked at "
            f"${prior_max:,.2f}, well under this one",
            f"The customer does pay '{txn.payee}' regularly, so the question is the size rather "
            f"than the destination. Check what made this instalment larger than the others.",
        )

    def _is_routine_for_payee(
        self, txn: Transaction, debit_txns: List[Transaction], observation_days: int
    ) -> bool:
        """
        True when this payee is a standing commitment of the customer's at
        roughly this size.

        Repetition alone is not enough. Three wires to the same stranger in three
        days repeat, but they are the very thing being looked for; a rent or a
        standing investment sweep recurs month after month. Requiring the
        payments to span time is what separates the two.
        """
        siblings = [t for t in debit_txns if t.payee == txn.payee and t is not txn]
        if len(siblings) < 2:
            return False

        dates = [datetime.strptime(t.date, "%Y-%m-%d") for t in siblings]
        dates.append(datetime.strptime(txn.date, "%Y-%m-%d"))
        required_span = min(
            self.RELATIONSHIP_MIN_DAYS,
            observation_days * self.RELATIONSHIP_WINDOW_FRACTION,
        )
        if (max(dates) - min(dates)).days < required_span:
            return False

        return txn.amount <= max(t.amount for t in siblings) * self.ESTABLISHED_PAYEE_TOLERANCE
