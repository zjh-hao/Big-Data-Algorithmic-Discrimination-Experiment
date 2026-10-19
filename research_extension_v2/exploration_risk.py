import numpy as np
from research_extension_v2.selective_label_ipw import validate_ledger


def risk_audit(ledger):
    validate_ledger(ledger)
    regular = ledger.proposed_approval == 1
    explored = (ledger.actual_approval == 1) & ~regular
    actual = ledger.actual_approval == 1
    def rate(m): return float(ledger.loc[m, 'outcome'].mean()) if m.any() else None
    return dict(regular_n=int(regular.sum()), regular_risk=rate(regular),
        exploration_n=int(explored.sum()), exploration_risk=rate(explored),
        actual_n=int(actual.sum()), actual_risk=rate(actual),
        actual_bad=int(ledger.loc[actual, 'outcome'].sum()))
