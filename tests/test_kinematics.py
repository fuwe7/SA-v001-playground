"""Тесты времени в воздухе и скорости выпуска."""
from __future__ import annotations

from core.kinematics import LiveKinematics, air_time, release_speed, speed_series


def test_air_time_detects_jump() -> None:
    # стойка y=0.9, прыжок y=0.7 на 3 кадрах (10 fps -> 0.3 c)
    times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ankle = [0.9, 0.7, 0.7, 0.7, 0.9, 0.9]
    assert air_time(times, ankle) == 0.3


def test_air_time_no_jump() -> None:
    times = [0.0, 0.1, 0.2]
    ankle = [0.9, 0.9, 0.9]
    assert air_time(times, ankle) == 0.0


def test_air_time_ignores_none() -> None:
    times = [0.0, 0.1, 0.2]
    ankle = [None, None, None]
    assert air_time(times, ankle) == 0.0


def test_speed_series() -> None:
    times = [0.0, 1.0, 2.0]
    pts = [(0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (3.0, 4.0, 0.0)]
    assert speed_series(times, pts) == [5.0, 0.0]


def test_release_speed_peak_in_window() -> None:
    times = [0.0, 1.0, 2.0, 3.0]
    pts = [(0.0, 0, 0), (1.0, 0, 0), (5.0, 0, 0), (5.0, 0, 0)]
    # скорости: 1, 4, 0 -> пик у выпуска (idx=2) = 4
    assert release_speed(times, pts, release_idx=3, window=3) == 4.0


def test_live_kinematics() -> None:
    kin = LiveKinematics()
    kin.update((0.0, 0.0, 0.0), 0.9, 0.0)   # стойка, baseline = 0.9
    kin.update((1.0, 0.0, 0.0), 0.7, 1.0)   # оторвался: таймер стартует
    kin.update((1.0, 0.0, 0.0), 0.7, 2.0)   # всё ещё в воздухе
    assert kin.rel_speed == 0.0
    assert kin.air_time == 1.0
