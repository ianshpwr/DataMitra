from .models import ActionType, ImpactLevel
from dataclasses import dataclass

@dataclass
class RuleMatch:
    action_type:    ActionType
    impact_level:   ImpactLevel
    effort_level:   ImpactLevel
    owner:          str
    kpi:            str
    priority_boost: float   # added to base priority score


# Rules: (insight_type, severity, keyword_in_title) → RuleMatch
ECOMMERCE_RULES: list[tuple[tuple, RuleMatch]] = [
    # Anomaly rules
    (("anomaly", "critical", None), RuleMatch(
        ActionType.investigate, ImpactLevel.high, ImpactLevel.low,
        "Data Engineering", "anomaly_resolution_rate", 0.3
    )),
    (("anomaly", "warning", None), RuleMatch(
        ActionType.investigate, ImpactLevel.medium, ImpactLevel.low,
        "Data Engineering", "anomaly_resolution_rate", 0.2
    )),

    # Null / data quality rules
    (("summary", "critical", "null"), RuleMatch(
        ActionType.fix, ImpactLevel.high, ImpactLevel.medium,
        "Data Engineering", "data_completeness_pct", 0.25
    )),
    (("summary", "warning", "null"), RuleMatch(
        ActionType.fix, ImpactLevel.medium, ImpactLevel.low,
        "Data Engineering", "data_completeness_pct", 0.15
    )),

    # Trend rules
    (("trend", "critical", "declin"), RuleMatch(
        ActionType.escalate, ImpactLevel.high, ImpactLevel.medium,
        "Product / Marketing", "monthly_revenue_growth", 0.35
    )),
    (("trend", "warning", "declin"), RuleMatch(
        ActionType.optimise, ImpactLevel.medium, ImpactLevel.medium,
        "Marketing", "conversion_rate", 0.2
    )),
    (("trend", "critical", "increas"), RuleMatch(
        ActionType.monitor, ImpactLevel.medium, ImpactLevel.low,
        "Operations", "fulfillment_rate", 0.1
    )),

    # Cancellation / status rules
    (("distribution", "warning", "cancel"), RuleMatch(
        ActionType.fix, ImpactLevel.high, ImpactLevel.medium,
        "Product / CX", "cancellation_rate", 0.3
    )),
    (("distribution", "warning", "pending"), RuleMatch(
        ActionType.investigate, ImpactLevel.medium, ImpactLevel.low,
        "Operations", "order_fulfillment_rate", 0.15
    )),

    # Correlation rules
    (("correlation", "info", None), RuleMatch(
        ActionType.optimise, ImpactLevel.medium, ImpactLevel.medium,
        "Analytics", "revenue_per_order", 0.05
    )),

    # Distribution / skew rules
    (("distribution", "info", None), RuleMatch(
        ActionType.monitor, ImpactLevel.low, ImpactLevel.low,
        "Analytics", "distribution_stability", 0.0
    )),
]

# Generic fallback rules by severity only
FALLBACK_RULES = {
    "critical": RuleMatch(ActionType.escalate,    ImpactLevel.high,   ImpactLevel.medium, "Leadership",       "business_health_score", 0.25),
    "warning":  RuleMatch(ActionType.investigate, ImpactLevel.medium, ImpactLevel.low,    "Operations",       "issue_resolution_rate", 0.10),
    "info":     RuleMatch(ActionType.monitor,     ImpactLevel.low,    ImpactLevel.low,    "Analytics",        "kpi_dashboard",         0.0),
}


def match_rule(
    insight_type: str,
    severity:     str,
    title:        str,
    domain:       str = "ecommerce",
) -> RuleMatch:
    title_lower = title.lower()
    rules = ECOMMERCE_RULES   # extend with FINANCE_RULES etc. later

    for (r_type, r_sev, r_keyword), match in rules:
        type_ok    = (r_type    == insight_type)
        sev_ok     = (r_sev     == severity)
        keyword_ok = (r_keyword is None) or (r_keyword in title_lower)
        if type_ok and sev_ok and keyword_ok:
            return match

    # Try without keyword
    for (r_type, r_sev, r_keyword), match in rules:
        if r_type == insight_type and r_sev == severity and r_keyword is None:
            return match

    return FALLBACK_RULES.get(severity, FALLBACK_RULES["info"])