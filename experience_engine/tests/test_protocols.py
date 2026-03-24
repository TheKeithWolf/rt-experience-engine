"""TEST-P1-047 through TEST-P1-048: Protocols and Range types."""

import pytest

from ..pipeline.protocols import (
    BoardFiller,
    Range,
    RangeFloat,
    SpatialSolver,
)


def test_p1_047_protocols_are_runtime_checkable() -> None:
    """TEST-P1-047: Protocols are runtime-checkable."""

    class MockSolver:
        def solve(self, plan, gravity_dag, variance):
            return None

    class MockFiller:
        def fill(self, board, pinned, constraints):
            return None

    assert isinstance(MockSolver(), SpatialSolver)
    assert isinstance(MockFiller(), BoardFiller)


def test_p1_048_range_rejects_min_gt_max() -> None:
    """TEST-P1-048: Range rejects min > max."""
    with pytest.raises(ValueError, match="min_val.*max_val"):
        Range(5, 3)

    with pytest.raises(ValueError, match="min_val.*max_val"):
        RangeFloat(2.5, 1.0)

    # Valid ranges should work
    r = Range(3, 5)
    assert r.contains(4)
    assert not r.contains(6)

    rf = RangeFloat(1.0, 2.5)
    assert rf.contains(1.5)
    assert not rf.contains(3.0)
