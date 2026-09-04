"""Data Models - Pydantic Schemas for Transaction Analysis"""

from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import List, Optional, Set
import uuid


class Transaction(BaseModel):
    """
    Individual transaction in customer history.

    Accepts the input field name "type" (as it appears in the source history)
    and exposes it as `transaction_type` internally.
    """
    model_config = ConfigDict(populate_by_name=True)

    date: str                                   # "YYYY-MM-DD" (mandatory)
    payee: str                                  # Entity receiving/sending money
    amount: float                               # Positive number
    channel: str                                # "Debit Card", "ACH", "Wire", "Online", ...
    transaction_type: str = Field(alias="type")  # "debit" (outflow) or "credit" (inflow)
    description: str = ""                       # Free-text label; optional in source data
    timestamp: Optional[str] = None             # "HH:MM" when time-of-day is known
    transaction_id: Optional[str] = None        # Auto-generated if not provided

    def __init__(self, **data):
        super().__init__(**data)
        if not self.transaction_id:
            self.transaction_id = f"TXN_{uuid.uuid4().hex[:8].upper()}"


class CustomerHistory(BaseModel):
    """Complete transaction history for a customer"""
    customer_id: str
    account_type: str
    transactions: List[Transaction]


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
    risk_level: str
    risk_score: float
    summary: str
    findings: Optional[List[Finding]] = None
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
