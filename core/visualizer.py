"""Визуализация анализа броска: видео + скелет + панель статистики.

Построено на том же ядре (PoseDetector + ShotDetector), что и headless-пайплайн,
поэтому логика расчётов не дублируется.

Два режима использования:
- live-окна (`run_visualization(..., show=True)`) — как в старом main.py: окно с
  видео/скелетом + окно "Live Stats";
- запись аннотированного видео (`save_path=...`) — кадр «видео + панель статистики»
  склеивается и пишется в файл (удобно делиться и работает без дисплея).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.biomechanics import compute_joint_angles
from core.detector import PoseDetector
from core.kinematics import LiveKinematics, ankle_level, wrist_world
from core.state_machine import ShotDetector
from core.view import detect_view

# Связи скелета MediaPipe Pose (туловище + обе руки + обе ноги)
SKELETON_LINKS = [
    (11, 12), (11, 23), (12, 24), (23, 24),     # туловище
    (12, 14), (14, 16),                         # правая рука
    (11, 13), (13, 15),                         # левая рука
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),  # правая нога
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),  # левая нога
]

R_SHOULDER, R_ELBOW, R_WRIST = 12, 14, 16
R_HIP, R_KNEE, R_ANKLE = 24, 26, 28
R_HEEL, R_FOOT_INDEX = 30, 32
L_HEEL, L_FOOT_INDEX = 29, 31
POSE_LANDMARK_COUNT = 33

STATS_WIDTH = 400


def _fmt_deg(value) -> str:
    """Форматирует угол для панели (Hershey-шрифт OpenCV не умеет кириллицу)."""
    if value is None:
        return "N/A"
    return f"{int(round(value))} deg"


def draw_skeleton(img, lm_list) -> None:
    """Рисует полный скелет (обе стороны тела)."""
    if not lm_list:
        return
    for p1, p2 in SKELETON_LINKS:
        if p1 < len(lm_list) and p2 < len(lm_list):
            x1, y1 = lm_list[p1][1], lm_list[p1][2]
            x2, y2 = lm_list[p2][1], lm_list[p2][2]
            cv2.line(img, (x1, y1), (x2, y2), (240, 240, 240), 2, cv2.LINE_AA)
    for idx in range(11, 33):
        if idx < len(lm_list):
            cx, cy = lm_list[idx][1], lm_list[idx][2]
            color = (255, 100, 0) if idx % 2 == 0 else (0, 100, 255)
            cv2.circle(img, (cx, cy), 4, color, -1)


def render_stats_panel(metrics: dict, height: int, width: int = STATS_WIDTH):
    """Рисует панель «Live Stats» с метриками (раскладка подстраивается под число метрик)."""
    board = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(board, (0, 0), (width, 70), (30, 30, 30), -1)
    cv2.putText(board, "LIVE STATS", (20, 45), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 1)

    top = 95
    n = max(1, len(metrics))
    step = min(76, max(46, (height - top - 16) // n))
    val_scale = 0.9 if step >= 70 else 0.7

    y = top
    for key, value in metrics.items():
        cv2.putText(board, str(key), (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        val_str = str(value)
        color = (255, 255, 255)
        if any(w in val_str for w in ("Great", "Good", "PERFECT", "RELEASE")):
            color = (0, 255, 0)
        elif any(w in val_str for w in ("Short", "Bad", "No Legs")):
            color = (0, 0, 255)
        elif any(w in val_str for w in ("PREP", "Load")):
            color = (0, 255, 255)
        elif val_str == "N/A":
            color = (120, 120, 120)
        cv2.putText(board, val_str, (20, y + 26), cv2.FONT_HERSHEY_SIMPLEX, val_scale, color, 2)
        y += step
        cv2.line(board, (20, y - 18), (width - 20, y - 18), (50, 50, 50), 1)
    return board


def annotate_frame(detector: PoseDetector, logic: ShotDetector, img,
                   kin: "LiveKinematics | None" = None, t: float = 0.0):
    """Обрабатывает один кадр: детекция позы, скелет, метрики, обновление стейт-машины.

    Возвращает (annotated_img, metrics).
    """
    detector.find_pose(img, draw=False)
    lm_list = detector.find_position(img, draw=False)

    metrics = {"STATUS": "Waiting", "SHOTS": logic.shot_count, "VIEW": "--",
               "R ELBOW": "--", "L ELBOW": "--", "R KNEE": "--",
               "L KNEE": "--", "TORSO LEAN": "--",
               "AIR TIME": "--", "REL SPEED": "--"}

    if len(lm_list) >= POSE_LANDMARK_COUNT:
        draw_skeleton(img, lm_list)

        elbow = detector.find_angle(img, R_SHOULDER, R_ELBOW, R_WRIST, draw=False)
        knee = detector.find_angle(img, R_HIP, R_KNEE, R_ANKLE, draw=False)
        r_foot = detector.find_vector_angle(R_HEEL, R_FOOT_INDEX)
        l_foot = detector.find_vector_angle(L_HEEL, L_FOOT_INDEX)
        feet_diff = abs(abs(r_foot) - abs(l_foot))

        shoulder_y = lm_list[R_SHOULDER][2]
        wrist_y = lm_list[R_WRIST][2]
        logic.check_shot(elbow, knee, feet_diff, wrist_y, shoulder_y)

        world_lm = detector.find_world_position()
        angles = compute_joint_angles(world_lm)
        metrics = {
            "STATUS": logic.feedback or logic.state,
            "SHOTS": logic.shot_count,
            "VIEW": detect_view(world_lm),
            "R ELBOW": _fmt_deg(angles["right_elbow"]),
            "L ELBOW": _fmt_deg(angles["left_elbow"]),
            "R KNEE": _fmt_deg(angles["right_knee"]),
            "L KNEE": _fmt_deg(angles["left_knee"]),
            "TORSO LEAN": _fmt_deg(angles["torso_lean"]),
            "AIR TIME": "--",
            "REL SPEED": "--",
        }
        if kin is not None:
            kin.update(wrist_world(world_lm), ankle_level(lm_list, img.shape[0]), t)
            metrics["AIR TIME"] = f"{kin.air_time:.2f} s"
            metrics["REL SPEED"] = f"{kin.rel_speed:.1f} m/s"

    cv2.putText(img, f"Shots: {logic.shot_count}", (30, 60),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 100, 0), 3)
    return img, metrics


def _compose(img, stats_panel):
    """Склеивает кадр видео и панель статистики по горизонтали (для записи в файл)."""
    h = img.shape[0]
    if stats_panel.shape[0] != h:
        stats_panel = cv2.resize(stats_panel, (stats_panel.shape[1], h))
    return np.hstack([img, stats_panel])


def run_visualization(video_path, show: bool = True, save_path: str | None = None) -> ShotDetector:
    """Прогоняет видео с визуализацией.

    Args:
        video_path: путь к видео.
        show: показывать live-окна (видео + Live Stats). Требует GUI-сборку OpenCV.
        save_path: если задан — писать аннотированное видео (видео + панель) в файл.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay_ms = max(1, int(1000 / fps))

    detector = PoseDetector()
    logic = ShotDetector()
    kin = LiveKinematics()
    writer = None
    frame_idx = 0

    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break

            img, metrics = annotate_frame(detector, logic, img, kin, frame_idx / fps)
            frame_idx += 1
            stats_panel = render_stats_panel(metrics, height=img.shape[0])

            if save_path is not None:
                composed = _compose(img, stats_panel)
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(save_path, fourcc, fps,
                                             (composed.shape[1], composed.shape[0]))
                writer.write(composed)

            if show:
                cv2.imshow("Shot Analyzer", img)
                cv2.imshow("Live Stats", stats_panel)
                if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    return logic
