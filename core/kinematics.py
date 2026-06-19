"""Время в воздухе и скорость выпуска (Этап 1).

- **Время в воздухе** считаем по 2D-координатам стоп в кадре: при прыжке лодыжки
  поднимаются над уровнем стойки. Нужны именно пиксельные координаты (а не world
  landmarks, которые центрированы по бёдрам и не видят глобальный подъём тела).
- **Скорость выпуска** — скорость кисти бросающей руки по 3D world landmarks
  (метры, в системе тела): отражает резкость разгибания руки в момент выпуска.
"""
from __future__ import annotations

import math

R_WRIST = 16
L_ANKLE, R_ANKLE = 27, 28

DEFAULT_RISE_FRAC = 0.04
DEFAULT_RELEASE_WINDOW = 3


def ankle_level(lm_list, frame_h: int):
    """Средний нормированный y лодыжек (0..1, больше = ниже в кадре). None если нет точек."""
    if frame_h <= 0 or len(lm_list) <= max(L_ANKLE, R_ANKLE):
        return None
    return (lm_list[L_ANKLE][2] + lm_list[R_ANKLE][2]) / (2.0 * frame_h)


def wrist_world(world_lm_list):
    """3D-координаты бросающей кисти (x, y, z) или None."""
    for row in world_lm_list:
        if row[0] == R_WRIST:
            return (row[1], row[2], row[3])
    return None


def air_time(times: list[float], ankle_y: list[float], rise_frac: float = DEFAULT_RISE_FRAC) -> float:
    """Длительность самого долгого непрерывного «в воздухе» интервала (сек).

    Уровень стойки = максимум ankle_y (тело ниже всего). В воздухе — когда лодыжки
    поднялись выше этого уровня более чем на rise_frac высоты кадра.
    """
    samples = [(t, y) for t, y in zip(times, ankle_y) if y is not None]
    if len(samples) < 2:
        return 0.0
    baseline = max(y for _, y in samples)

    best = 0.0
    start = None
    for i, (t, y) in enumerate(samples):
        airborne = (baseline - y) > rise_frac
        if airborne and start is None:
            start = t
        if (not airborne or i == len(samples) - 1) and start is not None:
            end = t if not airborne else samples[i][0]
            best = max(best, end - start)
            start = None
    return round(best, 3)


def speed_series(times: list[float], pts: list) -> list[float]:
    """Модуль скорости между соседними кадрами (единицы pts в секунду)."""
    out: list[float] = []
    for i in range(1, len(times)):
        p0, p1 = pts[i - 1], pts[i]
        dt = times[i] - times[i - 1]
        if p0 is None or p1 is None or dt <= 0:
            out.append(0.0)
            continue
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p0)))
        out.append(dist / dt)
    return out


def release_speed(
    times: list[float],
    wrist_pts: list,
    release_idx: int,
    window: int = DEFAULT_RELEASE_WINDOW,
) -> float:
    """Пиковая скорость кисти (м/с) в окне кадров вокруг момента выпуска."""
    speeds = speed_series(times, wrist_pts)
    if not speeds:
        return 0.0
    hi = min(len(speeds), release_idx)
    lo = max(0, hi - window)
    chunk = speeds[lo:hi] or speeds[max(0, hi - 1):hi]
    return round(max(chunk), 2) if chunk else 0.0


class LiveKinematics:
    """Потоковый счётчик для визуализации: мгновенная скорость кисти и таймер прыжка."""

    def __init__(self) -> None:
        self._prev_wrist = None
        self._prev_t = None
        self._baseline = None
        self._air_start = None
        self.rel_speed = 0.0
        self.air_time = 0.0

    def update(self, wrist, ankle_y, t: float) -> None:
        if wrist is not None and self._prev_wrist is not None and self._prev_t is not None:
            dt = t - self._prev_t
            if dt > 0:
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(wrist, self._prev_wrist)))
                self.rel_speed = dist / dt
        self._prev_wrist = wrist
        self._prev_t = t

        if ankle_y is not None:
            self._baseline = ankle_y if self._baseline is None else max(self._baseline, ankle_y)
            airborne = (self._baseline - ankle_y) > DEFAULT_RISE_FRAC
            if airborne:
                if self._air_start is None:
                    self._air_start = t
                self.air_time = t - self._air_start
            else:
                self._air_start = None
