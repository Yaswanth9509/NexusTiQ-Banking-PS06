"""Data Models - Pydantic Schemas for Transaction Analysis"""

from pydantic import BaseModel, Field, ConfigDict, model_validator
from datetime import datetime
from typing import List, Optional, Set

from src.utils import make_transaction_id


class Transaction(BaseModel):
    """
    Individual transaction in customer history.

    Accepts the input field name "type" (as it appears in the source history)
    and exposes it as `transaction_type` internally.
    """
    model_config = ConfigDict(populate_by_name=True)

    date: str                                   # "YYYY-MM-DD" (mandatory)
    payee: str                                  # Entity receiving/sending money
    amount: float = Field(ge=0)                 # Positive; direction is in transaction_type
    channel: str                                # "Debit Card", "ACH", "Wire", "Online", ...
    transaction_type: str = Field(alias="type")  # "debit" (outflow) or "credit" (inflow)
    description: str = ""                       # Free-text label; optional in source data
    timestamp: Optional[str] = None             # "HH:MM" when time-of-day is known
    transaction_id: Optional[str] = None        # Auto-generated if not provided

    # transaction_id is filled in by CustomerHistory, which knows the customer
    # and the row's position and can therefore derive a stable value.


class CustomerHistory(BaseModel):
    """Complete transaction history for a customer"""
    customer_id: str
    account_type: str
    transactions: List[Transaction]

    @model_validator(mode="after")
    def _check_dates_are_parseable(self):
        """
        Reject a malformed date rather than carrying it into the analysis.

        Every rule reasons over time - windows, spans, ordering - so a date that
        cannot be read is not a field the system can shrug off. Failing here
        gives the caller a clear 422 naming the row, instead of a stack trace
        from somewhere deep in a rule.
        """
        for index, txn in enumerate(self.transactions):
            try:
                datetime.strptime(txn.date, "%Y-%m-%d")
            except (ValueError, TypeError):
                raise ValueError(
                    f"transaction {index} has an unreadable date {txn.date!r}; "
                    f"expected YYYY-MM-DD"
                )
        return self

    @model_validator(mode="after")
    def _assign_transaction_ids(self):
        """
        Give every transaction a stable identifier.

        Histories supplied by a caller may carry no ids at all; those in
        data/sample_customers.json already do, and are left alone. Either way
        the same transaction ends up with the same reference on every run, which
        is what lets a finding be followed back to the row it came from.
        """
        for index, txn in enumerate(self.transactions):
            if not txn.transaction_id:
                txn.transaction_id = make_transaction_id(
                    self.customer_id, index, txn.date, txn.payee, txn.amount
                )
        return self


class CustomerProfile(BaseModel):
    """Established baseline behavior profile"""
    customer_id: str
    typical_payees: Set[str]
    established_payees: Set[str]
    typical_channels: Set[str]
    median_amount: float
    median_debit_amount: float
    avg_amount: float
    min_amount: float
    max_amount: float
    std_dev_amount: float
    transaction_count: int
    observation_days: int
    avg_days_between_txns: float
    is_mature: bool
    maturity_level: str


class Finding(BaseModel):
    """A single triggered risk rule finding"""
    rule_triggered: str
    rule_weight: float
    transactions_involved: List[str]
    specific_details: str
    deviation_from_normal: str
    investigator_should_look: str
    confidence: float


class InvestigationReport(BaseModel):
    """Final output report to investigator"""
    risk_level: str          # "ROUTINE", "INVESTIGATE" or "ESCALATE"
    risk_score: float
    summary: str
    findings: Optional[List[Finding]] = None
    how_findings_connect: Optional[dict] = None
    customer_context: Optional[dict] = None
    recommendation: str
    timestamp: datetime
    source_customer_id: str
    analysis_confidence: float

    # --- Enrichment from the AI layer -------------------------------------
    # Everything below is additive. The risk level, the score and the findings
    # above are fixed by the rules before any model is consulted, and a report
    # is complete and valid with all three of these absent.
    destination_context: Optional[List[dict]] = None
    investigator_narrative: Optional[dict] = None
    ai_status: Optional[dict] = None
