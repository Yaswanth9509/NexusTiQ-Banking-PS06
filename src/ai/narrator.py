"""
Investigator narrative.

The rules decide what is true. This module asks a model to explain that
truth to the person who has to act on it, and then checks that it did not
add anything of its own on the way.

The separation matters. Nothing here can create, suppress or reweight a
finding - by the time this runs, the risk level and the score are already
fixed. What a model contributes is the part deterministic code is bad at:
saying plainly what changed about a customer's behaviour, what would explain
it innocently, and what to look at first.

Anything the model returns that cannot be traced back to the facts it was
given is discarded in full. A narrative citing a transaction that does not
exist would be worse than no narrative, because it reads exactly as
authoritative as a correct one.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from src.models import CustomerProfile, Finding

log = logging.getLogger(__name__)

TXN_ID_PATTERN = re.compile(r"TXN_[A-Z0-9]+")

SYSTEM_INSTRUCTION = """You write briefing notes for a bank's fraud investigators.

You are given findings that have already been established by deterministic
rules, together with the customer's baseline behaviour. Your job is to explain
them to an investigator who has thirty seconds to decide what to do next.

Hard constraints:
- Never state or imply that fraud has occurred, that a customer is guilty, or
  that a transaction is fraudulent. The findings are grounds for a question,
  not a conclusion.
- Use only the facts supplied. Do not introduce transactions, payees, amounts,
  dates or history that are not in the input. If you want to say something the
  facts do not support, leave it out.
- Refer to transactions by the exact identifiers given.
- Always offer the innocent explanation alongside the suspicious one. Most
  flagged accounts belong to people who have done nothing wrong.
- Write in plain, direct English. No hedging filler, no dramatic language."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {
            "type": "string",
            "description": "One sentence an investigator could read in a queue.",
        },
        "assessment": {
            "type": "string",
            "description": "Two to four sentences on what changed about this customer's behaviour.",
        },
        "where_to_start": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ordered, concrete first steps for the investigator.",
        },
        "innocent_explanations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ordinary explanations that would account for this activity.",
        },
        "not_established": {
            "type": "array",
            "items": {"type": "string"},
            "description": "What the transaction history cannot tell us either way.",
        },
    },
    "required": [
        "headline",
        "assessment",
        "where_to_start",
        "innocent_explanations",
        "not_established",
    ],
}


def build_prompt(
    customer_id: str,
    profile: CustomerProfile,
    findings: List[Finding],
    typology_notes: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Assemble the facts the model is allowed to reason over - and nothing else."""
    lines = [
        f"CUSTOMER: {customer_id}",
        "",
        "ESTABLISHED BASELINE",
        f"- History observed: {profile.transaction_count} transactions over {profile.observation_days} days",
        f"- Profile maturity: {profile.maturity_level}",
        f"- Median payment out: ${profile.median_debit_amount:,.2f}",
        f"- Payment range: ${profile.min_amount:,.2f} to ${profile.max_amount:,.2f}",
        f"- Regular payees: {', '.join(sorted(profile.typical_payees)[:10]) or 'none established'}",
        f"- Regular channels: {', '.join(sorted(profile.typical_channels)) or 'none established'}",
        "",
        "FINDINGS ESTABLISHED BY THE RULES",
    ]

    for index, finding in enumerate(findings, start=1):
        lines.extend([
            f"{index}. {finding.rule_triggered}",
            f"   What was found: {finding.specific_details}",
            f"   How it differs from baseline: {finding.deviation_from_normal}",
            f"   Transactions: {', '.join(finding.transactions_involved)}",
        ])

    if typology_notes:
        lines.extend(["", "DESTINATION REFERENCE (from the bank's typology document)"])
        for note in typology_notes:
            lines.append(
                f"- '{note['payee']}' resembles {note['label']} [{note['id']}], "
                f"posture {note['posture']}: {note['why_it_matters']}"
            )

    lines.extend([
        "",
        "Write the briefing note. Use only what appears above.",
    ])
    return "\n".join(lines)


def validate_grounding(
    narrative: Dict[str, Any],
    allowed_txn_ids: set,
    allowed_payees: set,
) -> Optional[str]:
    """
    Check the narrative introduced nothing of its own.

    Returns a reason string if the narrative should be rejected, or None if it
    is safe to publish.
    """
    body = " ".join(
        part if isinstance(part, str) else " ".join(part)
        for part in narrative.values()
        if isinstance(part, (str, list))
    )

    cited_ids = set(TXN_ID_PATTERN.findall(body))
    invented_ids = cited_ids - allowed_txn_ids
    if invented_ids:
        return f"cited transactions not present in the findings: {', '.join(sorted(invented_ids))}"

    # Payee names appear in the facts in single quotes; if the model quotes a
    # name that was never supplied, it invented a counterparty.
    quoted = {match.strip() for match in re.findall(r"'([^']{2,60})'", body)}
    invented_payees = {
        name for name in quoted
        if name not in allowed_payees and not name.startswith("TXN_")
    }
    if invented_payees:
        return f"referred to payees not present in the findings: {', '.join(sorted(invented_payees))}"

    if not narrative.get("headline") or not narrative.get("assessment"):
        return "narrative was incomplete"

    return None


async def generate_narrative(
    client,
    customer_id: str,
    profile: CustomerProfile,
    findings: List[Finding],
    typology_notes: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Produce the investigator narrative, or explain why there isn't one.

    Always returns a dict describing the outcome, so the report can be honest
    about whether a model contributed to it.
    """
    if not findings:
        return {"available": False, "reason": "no findings to narrate"}

    allowed_txn_ids = {
        txn_id for finding in findings for txn_id in finding.transactions_involved
    }
    allowed_payees = set(profile.typical_payees) | set(profile.typical_channels)
    for note in typology_notes or []:
        allowed_payees.add(note["payee"])
    # Payee names quoted inside the findings themselves are equally legitimate.
    for finding in findings:
        allowed_payees.update(
            re.findall(r"'([^']{2,60})'", finding.specific_details + finding.deviation_from_normal)
        )

    prompt = build_prompt(customer_id, profile, findings, typology_notes)
    narrative = await client.generate_json(
        prompt=prompt,
        schema=RESPONSE_SCHEMA,
        system_instruction=SYSTEM_INSTRUCTION,
    )

    if narrative is None:
        return {
            "available": False,
            "reason": f"model unavailable ({client.status.get('last_error') or 'unknown'})",
        }

    rejection = validate_grounding(narrative, allowed_txn_ids, allowed_payees)
    if rejection:
        log.warning("Rejected ungrounded narrative for %s: %s", customer_id, rejection)
        return {"available": False, "reason": f"narrative rejected - {rejection}"}

    narrative["available"] = True
    narrative["grounding"] = {
        "transactions_available_to_cite": sorted(allowed_txn_ids),
        "checked": "every cited transaction and payee was matched against the findings",
    }
    return narrative
