"""Тесты авто-определения ракурса."""
from __future__ import annotations

from core.view import VIEW_FRONT, VIEW_SIDE, ViewAggregator, detect_view


def _lm(idx, x, y, z, vis=1.0):
    return [idx, x, y, z, vis]


def _side_view():
    # плечи/бёдра разнесены по глубине (z), а не по горизонтали (x)
    return [
        _lm(11, 0.0, 1.0, 0.0), _lm(12, 0.0, 1.0, 0.3),
        _lm(23, 0.0, 0.0, 0.0), _lm(24, 0.0, 0.0, 0.3),
    ]


def _front_view():
    # плечи/бёдра разнесены по горизонтали (x)
    return [
        _lm(11, -0.2, 1.0, 0.0), _lm(12, 0.2, 1.0, 0.0),
        _lm(23, -0.15, 0.0, 0.0), _lm(24, 0.15, 0.0, 0.0),
    ]


def test_detect_side() -> None:
    assert detect_view(_side_view()) == VIEW_SIDE


def test_detect_front() -> None:
    assert detect_view(_front_view()) == VIEW_FRONT


def test_aggregator_majority() -> None:
    agg = ViewAggregator()
    for _ in range(5):
        agg.update(_side_view())
    agg.update(_front_view())
    view, conf = agg.result()
    assert view == VIEW_SIDE
    assert 0.0 < conf <= 1.0


def test_aggregator_empty() -> None:
    view, conf = ViewAggregator().result()
    assert conf == 0.0
