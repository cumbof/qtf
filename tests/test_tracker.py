"""Tests for LandscapeTracker."""

from qtf.core.tracker import LandscapeTracker


def test_initial_state():
    t = LandscapeTracker()
    assert t.history == []
    assert t.stage_markers == []
    assert t.current_iter == 0


def test_log_appends_energy():
    t = LandscapeTracker()
    t.log(10.0)
    t.log(8.0)
    assert t.history == [10.0, 8.0]


def test_log_increments_iter():
    t = LandscapeTracker()
    t.log(1.0)
    assert t.current_iter == 1
    t.log(2.0)
    assert t.current_iter == 2


def test_mark_stage_records_current_iter():
    t = LandscapeTracker()
    t.log(10.0)
    t.mark_stage("Stage1")
    assert t.stage_markers == [(1, "Stage1")]


def test_mark_stage_at_zero():
    t = LandscapeTracker()
    t.mark_stage("init")
    assert t.stage_markers == [(0, "init")]
    assert t.current_iter == 0  # mark_stage does not increment iter


def test_multiple_stages():
    t = LandscapeTracker()
    t.log(10.0)
    t.mark_stage("Stage1")
    t.log(8.0)
    t.mark_stage("Stage2")
    t.log(6.0)
    t.mark_stage("Stage3")
    assert t.stage_markers == [(1, "Stage1"), (2, "Stage2"), (3, "Stage3")]


def test_log_accepts_negative():
    t = LandscapeTracker()
    t.log(-99.5)
    assert t.history[0] == -99.5


def test_history_order_preserved():
    t = LandscapeTracker()
    values = [3.0, 1.0, 4.0, 1.5]
    for v in values:
        t.log(v)
    assert t.history == values
