"""Авто-определение ракурса камеры по 3D world landmarks (Этап 1).

Идея: линия между плечами (и между бёдрами) при съёмке спереди/сзади «развёрнута»
по горизонтали (ось x), а при съёмке сбоку — уходит в глубину (ось z). По соотношению
горизонтальной и глубинной составляющих определяем ракурс. Это нужно, чтобы честно
говорить, каким метрикам можно доверять (сагиттальные углы локтя/колена — сбоку,
симметрия плеч/стоп — спереди).
"""
from __future__ import annotations

L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24

VIEW_SIDE = "side"
VIEW_FRONT = "front/back"
VIEW_ANGLED = "angled"

SIDE_MAX_RATIO = 0.40
FRONT_MIN_RATIO = 0.65
DEFAULT_VIS_THRESHOLD = 0.5


def view_ratio(world_lm_list, vis_threshold: float = DEFAULT_VIS_THRESHOLD):
    """Доля горизонтального разворота корпуса: |dx| / (|dx| + |dz|).

    ~1.0 — корпус развёрнут к камере (вид спереди/сзади), ~0.0 — вид сбоку.
    Возвращает None, если плечи/бёдра недостаточно видны.
    """
    pts = {row[0]: (row[1], row[2], row[3], row[4]) for row in world_lm_list}
    needed = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    if any(i not in pts for i in needed):
        return None
    if min(pts[i][3] for i in needed) < vis_threshold:
        return None

    dx = abs(pts[R_SHOULDER][0] - pts[L_SHOULDER][0]) + abs(pts[R_HIP][0] - pts[L_HIP][0])
    dz = abs(pts[R_SHOULDER][2] - pts[L_SHOULDER][2]) + abs(pts[R_HIP][2] - pts[L_HIP][2])
    if dx + dz == 0.0:
        return None
    return dx / (dx + dz)


def classify_view(ratio) -> str:
    """Классифицирует ракурс по соотношению (см. view_ratio)."""
    if ratio is None:
        return VIEW_ANGLED
    if ratio < SIDE_MAX_RATIO:
        return VIEW_SIDE
    if ratio > FRONT_MIN_RATIO:
        return VIEW_FRONT
    return VIEW_ANGLED


def detect_view(world_lm_list, vis_threshold: float = DEFAULT_VIS_THRESHOLD) -> str:
    """Ракурс по одному кадру."""
    return classify_view(view_ratio(world_lm_list, vis_threshold))


class ViewAggregator:
    """Накапливает оценки ракурса по кадрам и выдаёт итоговый ракурс видео."""

    def __init__(self) -> None:
        self._ratios: list[float] = []

    def update(self, world_lm_list, vis_threshold: float = DEFAULT_VIS_THRESHOLD) -> None:
        ratio = view_ratio(world_lm_list, vis_threshold)
        if ratio is not None:
            self._ratios.append(ratio)

    def result(self) -> tuple[str, float]:
        """Возвращает (ракурс, уверенность 0..1).

        Уверенность — доля кадров, согласных с итоговым ракурсом.
        """
        if not self._ratios:
            return VIEW_ANGLED, 0.0
        labels = [classify_view(r) for r in self._ratios]
        winner = max(set(labels), key=labels.count)
        confidence = labels.count(winner) / len(labels)
        return winner, round(confidence, 2)
