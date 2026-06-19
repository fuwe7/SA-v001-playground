"""Тесты расчёта 3D-углов суставов."""
from __future__ import annotations

from core.biomechanics import compute_joint_angles


def _lm(idx: int, x: float, y: float, z: float, vis: float = 1.0):
    return [idx, x, y, z, vis]


def _right_arm_90deg(vis: float = 1.0):
    # R_SHOULDER=12, R_ELBOW=14, R_WRIST=16 — прямой угол в локте
    return [
        _lm(12, 0.0, 1.0, 0.0, vis),
        _lm(14, 0.0, 0.0, 0.0, vis),
        _lm(16, 1.0, 0.0, 0.0, vis),
    ]


def test_right_elbow_right_angle() -> None:
    angles = compute_joint_angles(_right_arm_90deg())
    assert angles["right_elbow"] == 90.0


def test_low_visibility_returns_none() -> None:
    angles = compute_joint_angles(_right_arm_90deg(vis=0.1))
    assert angles["right_elbow"] is None


def test_missing_landmarks_return_none() -> None:
    angles = compute_joint_angles([])  # точек нет вовсе
    assert angles["right_elbow"] is None
    assert angles["torso_lean"] is None
