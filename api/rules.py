"""
api/rules.py — Business Rules Engine

Evaluates transactions against strict business policies *before* or *alongside*
the ML model execution. In an enterprise system, Machine Learning is just one
signal; hard-coded rules govern legal compliance and specific business needs.

Examples:
- OFAC Sanctions (e.g. cannot process transactions to North Korea)
- VIP Allow-lists (high net worth individuals bypass ML friction)
- Hard card limits (transactions > $1,000,000 trigger manual review regardless of ML)
"""

from typing import Dict, Optional, Tuple

# OFAC (Office of Foreign Assets Control) comprehensively sanctioned countries
SANCTIONED_COUNTRIES = {"KP", "IR", "SY", "CU"}

# High risk countries (FATF grey/black lists)
HIGH_RISK_COUNTRIES = {"MM", "ML", "SN", "SS", "UG", "YE", "HT", "NG"}

# Customer IDs with "VIP" status (no friction unless critical risk)
VIP_CUSTOMERS = {"CUST_VIP_001", "CUST_VIP_002"}

# Transaction amount limit requiring mandatory review
MAX_AUTO_APPROVE_AMOUNT = 50_000.0


def evaluate_fraud_rules(transaction: Dict) -> Tuple[Optional[str], Optional[str], str]:
    """
    Evaluate a single transaction against business rules.
    
    Returns:
        (rule_label, rule_action, rule_name)
        If no rule is triggered, returns (None, None, "")
    """
    country = transaction.get("country", "").upper()
    amount = transaction.get("amount", 0.0)
    customer_id = transaction.get("customer_id", "")

    # 1. Legal / Sanctions (Absolute block)
    if country in SANCTIONED_COUNTRIES:
        return ("🚨 CRITICAL RISK", "Block immediately - OFAC Sanctions", "OFAC_SANCTIONS")

    # 2. Hard limits
    if amount > MAX_AUTO_APPROVE_AMOUNT:
        return ("🟠 MEDIUM RISK", "Flag for manual review - Limit Exceeded", "LARGE_AMOUNT_REVIEW")

    # 3. VIP Allow-list
    if customer_id in VIP_CUSTOMERS:
        return ("🟢 LEGITIMATE", "Approve VIP", "VIP_ALLOWLIST")

    # 4. FATF High Risk
    if country in HIGH_RISK_COUNTRIES and amount > 5000:
        return ("🔴 HIGH RISK", "Decline & Alert - High Risk Jurisdiction", "FATF_HIGH_RISK")

    return (None, None, "")


def override_ml_decision(
    ml_label: str, ml_action: str, ml_prob: float, 
    rule_label: Optional[str], rule_action: Optional[str], rule_name: str
) -> Tuple[str, str, str]:
    """
    Merge the ML model's decision with the Business Rules Engine decision.
    Rules take precedence for blocking or VIP overrides, while ML handles
    complex behavioral anomalies.
    """
    # If no rule triggered, use ML
    if not rule_label:
        return ml_label, ml_action, "ML_MODEL"
    
    # VIPs bypass everything EXCEPT ML "CRITICAL RISK"
    if rule_name == "VIP_ALLOWLIST":
        if "CRITICAL" in ml_label:
            return ml_label, "Review VIP (ML Critical)", "ML_MODEL_VIP_OVERRIDE"
        return rule_label, rule_action, rule_name
    
    # OFAC Sanctions block absolutely everything
    if rule_name == "OFAC_SANCTIONS":
        return rule_label, rule_action, rule_name
        
    # For anything else, return the MORE SEVERE action
    severity = {
        "🟢 LEGITIMATE": 0,
        "🟡 LOW RISK": 1,
        "🟠 MEDIUM RISK": 2,
        "🔴 HIGH RISK": 3,
        "🚨 CRITICAL RISK": 4,
        "🚨 HIGH AML RISK": 4,
    }
    
    ml_sev = severity.get(ml_label, 0)
    rule_sev = severity.get(rule_label, 0)
    
    if rule_sev > ml_sev:
        return rule_label, rule_action, rule_name
    return ml_label, ml_action, "ML_MODEL"
