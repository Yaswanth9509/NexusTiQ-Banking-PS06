"""Base Risk Rule Class - Abstract Base for All Rules"""

from abc import ABC, abstractmethod
from typing import List, Optional
from src.models import Transaction, CustomerProfile, Finding


class RiskRule(ABC):
    """Abstract base class for risk rules"""

    def __init__(self, name: str, description: str, risk_weight: float):
        self.name = name
        self.description = description
        self.risk_weight = risk_weight

    @abstractmethod
    def check(
        self, transactions: List[Transaction], profile: CustomerProfile
    ) -> Optional[Finding]:
        """
        Check if rule is triggered

        Returns:
            Finding if rule triggered, None otherwise
        """
        pass
