"""Расчёт углов суставов по 3D world landmarks (Этап 1).

Углы считаются по трёхмерным координатам MediaPipe (origin — центр бёдер, метры),
поэтому устойчивы к ракурсу камеры: угол между двумя костями не зависит от того,
под каким углом стоит камера, в отличие от плоских пиксельных углов.

Покрывается полный набор суставов обеих сторон + наклон корпуса.
"""
from __future__ import annotations

import math

L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

DEFAULT_VIS_THRESHOLD = 0.5


def _angle_3d(a: tuple, b: tuple, c: tuple) -> float:
    """Угол в точке b (в градусах) между лучами b->a и b->c."""
    ba = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    na = math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2)
    nc = math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2)
    if na == 0.0 or nc == 0.0:
        return 0.0
    cos = max(-1.0, min(1.0, dot / (na * nc)))
    return math.degrees(math.acos(cos))


def compute_joint_angles(world_lm_list, vis_threshold: float = DEFAULT_VIS_THRESHOLD) -> dict:
    """Считает углы ключевых суставов по 3D world landmarks.

    Args:
        world_lm_list: список [id, x, y, z, visibility] из PoseDetector.find_world_position().
        vis_threshold: минимальная видимость всех трёх точек, иначе угол = None.

    Returns:
        Словарь {имя_сустава: угол в градусах или None}. Также ключ "torso_lean" —
        отклонение корпуса от вертикали (0 = прямой корпус), приблизительно.
    """
    pts = {row[0]: (row[1], row[2], row[3], row[4]) for row in world_lm_list}

    def angle(a: int, b: int, c: int):
        if a not in pts or b not in pts or c not in pts:
            return None
        if min(pts[a][3], pts[b][3], pts[c][3]) < vis_threshold:
            return None
        return round(_angle_3d(pts[a][:3], pts[b][:3], pts[c][:3]), 1)

    angles = {
        "right_elbow": angle(R_SHOULDER, R_ELBOW, R_WRIST),
        "left_elbow": angle(L_SHOULDER, L_ELBOW, L_WRIST),
        "right_knee": angle(R_HIP, R_KNEE, R_ANKLE),
        "left_knee": angle(L_HIP, L_KNEE, L_ANKLE),
        "right_shoulder": angle(R_ELBOW, R_SHOULDER, R_HIP),
        "left_shoulder": angle(L_ELBOW, L_SHOULDER, L_HIP),
        "right_hip": angle(R_SHOULDER, R_HIP, R_KNEE),
        "left_hip": angle(L_SHOULDER, L_HIP, L_KNEE),
    }
    angles["torso_lean"] = _torso_lean(pts, vis_threshold)
    return angles


def _torso_lean(pts: dict, vis_threshold: float):
    """Отклонение линии бёдра→плечо от вертикальной оси, в градусах (приблизительно)."""
    needed = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    if any(i not in pts for i in needed):
        return None
    if min(pts[i][3] for i in needed) < vis_threshold:
        return None

    shoulder_c = _midpoint(pts[L_SHOULDER], pts[R_SHOULDER])
    hip_c = _midpoint(pts[L_HIP], pts[R_HIP])
    v = (shoulder_c[0] - hip_c[0], shoulder_c[1] - hip_c[1], shoulder_c[2] - hip_c[2])
    norm = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if norm == 0.0:
        return None
    # y-ось — вертикаль в системе world landmarks; отклонение корпуса от неё
    cos = max(-1.0, min(1.0, abs(v[1]) / norm))
    lean = math.degrees(math.acos(cos))
    return round(lean, 1)


def _midpoint(p1: tuple, p2: tuple) -> tuple:
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0, (p1[2] + p2[2]) / 2.0)
