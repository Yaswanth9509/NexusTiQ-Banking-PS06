"""TransactionAnalyzer - Main System Coordinator"""

from datetime import datetime
from statistics import stdev
from typing import List, Optional
from src.models import (
    Transaction,
    CustomerHistory,
    CustomerProfile,
    InvestigationReport,
    Finding,
)
from src.rules.base import RiskRule


class TransactionAnalyzer:
    """Main system coordinator for transaction analysis"""

    # The heaviest weight any single rule carries; used to normalise scoring.
    MAX_RULE_WEIGHT = 0.40

    # At or above this score the report is escalated rather than merely flagged.
    # Only three or more independent findings can reach it.
    ESCALATION_THRESHOLD = 0.80

    def __init__(self, risk_rules: List[RiskRule]):
        self.risk_rules = risk_rules

    async def analyze(self, customer_history: CustomerHistory) -> InvestigationReport:
        """
        Main entry point for analyzing customer transaction history

        Args:
            customer_history: Customer's transaction history

        Returns:
            InvestigationReport with findings and risk assessment
        """
        if not customer_history.transactions:
            raise ValueError("No transactions provided")

        # Step 1: Build customer profile
        profile = self.build_customer_profile(customer_history)

        # Step 2: Check profile maturity
        if profile.maturity_level == "INSUFFICIENT":
            return InvestigationReport(
                risk_level="ROUTINE",
                risk_score=0.0,
                summary="Insufficient transaction history for proper analysis. Minimum 10 transactions and 3+ days required.",
                findings=None,
                customer_context={"transaction_count": len(customer_history.transactions), "observation_days": profile.observation_days},
                recommendation="Request extended history for reliable analysis",
                timestamp=datetime.now(),
                source_customer_id=customer_history.customer_id,
                analysis_confidence=0.2
            )

        # Step 3: Check all rules, holding back the ones a forming profile
        # cannot support. PATTERN_BREAK compares a customer against their own
        # past; with under a month observed there is not yet a past to compare to.
        findings = []
        for rule in self.risk_rules:
            if profile.maturity_level == "EMERGING" and rule.name == "PATTERN_BREAK":
                continue

            result = rule.check(customer_history.transactions, profile)
            if result:
                findings.append(result)

        # Step 4: Calculate risk score
        risk_score = self._calculate_risk_score(findings)

        # Step 5: Generate report
        if findings:
            # The top band is named in risk_level itself rather than left for a
            # caller to infer by comparing the score against a threshold or
            # reading the recommendation prose. Three independent rules firing on
            # one customer is a different instruction to a fraud desk than one.
            if risk_score >= self.ESCALATION_THRESHOLD:
                risk_level = "ESCALATE"
                recommendation = (
                    "ESCALATE - several independent signals on the same customer. "
                    "Put this in front of an investigator now."
                )
                analysis_confidence = 0.95
            else:
                risk_level = "INVESTIGATE"
                recommendation = (
                    "INVESTIGATE - one or more signals worth a closer look. "
                    "Review when the queue allows."
                )
                analysis_confidence = 0.70

            report = InvestigationReport(
                risk_level=risk_level,
                risk_score=risk_score,
                summary=f"Detected {len(findings)} risk signal(s) - {', '.join([f.rule_triggered for f in findings])}",
                findings=findings,
                customer_context={
                    "maturity": profile.maturity_level,
                    "transaction_count": len(customer_history.transactions),
                    "observation_days": profile.observation_days,
                    "median_transaction": f"${profile.median_amount:.2f}",
                    "typical_payees_sample": list(profile.typical_payees)[:5],
                    "typical_channels_sample": list(profile.typical_channels)[:3]
                },
                recommendation=recommendation,
                timestamp=datetime.now(),
                source_customer_id=customer_history.customer_id,
                analysis_confidence=analysis_confidence
            )
        else:
            report = InvestigationReport(
                risk_level="ROUTINE",
                risk_score=0.0,
                summary="No suspicious signals detected. Account activity within normal parameters.",
                findings=None,
                customer_context={
                    "maturity": profile.maturity_level,
                    "transaction_count": len(customer_history.transactions),
                    "observation_days": profile.observation_days,
                    "median_transaction": f"${profile.median_amount:.2f}",
                    "analysis_basis": "All transactions consistent with established patterns"
                },
                recommendation="APPROVE - No action required. Activity appears routine.",
                timestamp=datetime.now(),
                source_customer_id=customer_history.customer_id,
                analysis_confidence=0.90
            )

        return report

    def build_customer_profile(
        self, customer_history: CustomerHistory
    ) -> CustomerProfile:
        """
        Build baseline customer behavior profile with maturity assessment

        Returns:
            CustomerProfile with established patterns
        """
        transactions = customer_history.transactions

        if len(transactions) == 0:
            raise ValueError("No transactions provided")

        # AMOUNT ANALYSIS
        #
        # Debits get their own median. Comparing a payment out against a figure
        # that includes salary credits measures nothing useful - the credits sit
        # an order of magnitude above everyday spending and drag the reference off.
        amounts = [t.amount for t in transactions]
        amounts_sorted = sorted(amounts)
        median_amount = amounts_sorted[len(amounts_sorted) // 2]

        debit_amounts = sorted(t.amount for t in transactions if t.transaction_type == "debit")
        median_debit_amount = (
            debit_amounts[len(debit_amounts) // 2] if debit_amounts else median_amount
        )

        avg_amount = sum(amounts) / len(amounts)
        min_amount = min(amounts)
        max_amount = max(amounts)
        std_dev_amount = stdev(amounts) if len(amounts) > 1 else 0.0

        # PAYEE ANALYSIS (establish "typical" payees)
        payee_counts = {}
        for t in transactions:
            payee_counts[t.payee] = payee_counts.get(t.payee, 0) + 1
        typical_payees = {p for p, count in payee_counts.items() if count >= 2}

        # CHANNEL ANALYSIS
        channel_counts = {}
        for t in transactions:
            channel_counts[t.channel] = channel_counts.get(t.channel, 0) + 1
        typical_channels = {c for c, count in channel_counts.items() if count >= 2}

        # TIMING ANALYSIS
        # Derived from the full set rather than the first and last entries, so a
        # history that arrives out of order still yields the right window.
        parsed_dates = [datetime.strptime(t.date, "%Y-%m-%d") for t in transactions]
        observation_days = (max(parsed_dates) - min(parsed_dates)).days
        avg_days_between = (
            observation_days / (len(transactions) - 1) if len(transactions) > 1 else 0.0
        )

        # PROFILE MATURITY DETERMINATION
        #
        # Maturity governs how hard the rules are allowed to bite. A month of
        # observation is what makes a monthly rhythm - salary, rent, utilities -
        # visible at all, so it is the dividing line between a forming profile
        # and an established one.
        if len(transactions) < 10 or observation_days < 3:
            maturity_level = "INSUFFICIENT"
            is_mature = False
        elif observation_days < 30:
            maturity_level = "EMERGING"
            is_mature = False
        else:
            maturity_level = "ESTABLISHED"
            is_mature = True

        return CustomerProfile(
            customer_id=customer_history.customer_id,
            typical_payees=typical_payees,
            established_payees=typical_payees,
            typical_channels=typical_channels,
            median_amount=median_amount,
            median_debit_amount=median_debit_amount,
            avg_amount=avg_amount,
            min_amount=min_amount,
            max_amount=max_amount,
            std_dev_amount=std_dev_amount,
            transaction_count=len(transactions),
            observation_days=observation_days,
            avg_days_between_txns=avg_days_between,
            is_mature=is_mature,
            maturity_level=maturity_level
        )

    def _calculate_risk_score(self, findings: List[Finding]) -> float:
        """
        Calculate overall risk score from findings

        Formula:
        - 0 findings: 0.0 (ROUTINE)
        - 1 finding: 0.3 + (rule_weight * 0.3) = 0.4-0.6
        - 2 findings: 0.6 + (avg_weight * 0.15) = 0.65-0.75
        - 3+ findings: 0.8 + (avg_weight * 0.2) = 0.8-1.0

        Args:
            findings: List of triggered findings

        Returns:
            risk_score between 0.0 and 1.0
        """
        if len(findings) == 0:
            return 0.0

        # Rule weights run 0.10 to 0.40, so they are expressed as a fraction of
        # the heaviest rule before being applied. Used raw they would only ever
        # move the score by a few hundredths and every band would collapse onto
        # its floor - a single burst finding and a single odd-hours finding would
        # come out barely distinguishable.
        avg_rule_weight = sum(f.rule_weight for f in findings) / len(findings)
        weight_fraction = min(1.0, avg_rule_weight / self.MAX_RULE_WEIGHT)

        if len(findings) == 1:
            score = 0.30 + (weight_fraction * 0.30)   # 0.30 - 0.60
        elif len(findings) == 2:
            score = 0.60 + (weight_fraction * 0.15)   # 0.60 - 0.75
        else:
            score = 0.80 + (weight_fraction * 0.20)   # 0.80 - 1.00

        return round(min(1.0, max(0.0, score)), 2)
