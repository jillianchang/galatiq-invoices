from src.gate import compute_gate
from src.rules.codes import finding
from src.schemas import Gate


def f(code):
    return finding(code, code)


def test_no_blocking_findings_is_eligible():
    assert compute_gate([]) == Gate.ELIGIBLE
    assert (
        compute_gate([f("over_threshold"), f("near_threshold"), f("reprocessed_submission")])
        == Gate.ELIGIBLE
    )


def test_precedence():
    assert compute_gate([f("unknown_item"), f("total_mismatch")]) == Gate.RETURN_TO_VENDOR
    assert compute_gate([f("total_mismatch"), f("missing_vendor")]) == Gate.REJECT
    assert compute_gate([f("missing_vendor"), f("duplicate")]) == Gate.DUPLICATE


def test_gate_without_duplicate_shows_underlying_result():
    assert compute_gate([f("duplicate"), f("unknown_item")], ignore_duplicate=True) == Gate.HOLD
