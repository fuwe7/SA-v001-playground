"""Единый модуль анализа броска (источник правды).

Этап 0: headless-пайплайн «видео -> метрики». Без GUI-окон и без рисования,
чтобы его можно было одинаково использовать из CLI, тестов и (позже) из веб-интерфейса.

Логика расчёта углов и фаз броска переиспользует core.detector и core.state_machine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2

from core.biomechanics import compute_joint_angles
from core.detector import PoseDetector
from core.kinematics import air_time, ankle_level, release_speed, wrist_world
from core.state_machine import ShotDetector
from core.view import ViewAggregator

# Индексы ключевых точек MediaPipe Pose (правая сторона — основная для броска)
R_SHOULDER, R_ELBOW, R_WRIST = 12, 14, 16
R_HIP, R_KNEE, R_ANKLE = 24, 26, 28
R_HEEL, R_FOOT_INDEX = 30, 32
L_HEEL, L_FOOT_INDEX = 29, 31

# MediaPipe Pose всегда возвращает 33 точки, если поза найдена
POSE_LANDMARK_COUNT = 33


@dataclass
class AnalysisResult:
    """Структурированный результат анализа одного видео."""

    video_path: str
    frames_total: int
    frames_with_pose: int
    shots: int
    session_data: list
    view: str
    view_confidence: float
    air_time: float

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_video(
    video_path: str | Path,
    on_progress: Callable[[int], None] | None = None,
) -> AnalysisResult:
    """Прогоняет видео через детектор позы и стейт-машину броска.

    Args:
        video_path: путь к видеофайлу (mp4/mov/...).
        on_progress: необязательный колбэк, получает номер обработанного кадра.

    Returns:
        AnalysisResult с количеством бросков и метриками по каждому из них.

    Raises:
        FileNotFoundError: если видео не удалось открыть.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    detector = PoseDetector()
    logic = ShotDetector()
    view_agg = ViewAggregator()

    frames_total = 0
    frames_with_pose = 0
    times: list[float] = []
    ankle_ys: list = []
    wrists: list = []

    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames_total += 1

            detector.find_pose(img, draw=False)
            lm_list = detector.find_position(img, draw=False)

            if len(lm_list) >= POSE_LANDMARK_COUNT:
                frames_with_pose += 1
                world_lm = detector.find_world_position()
                view_agg.update(world_lm)

                times.append(frames_total / fps)
                ankle_ys.append(ankle_level(lm_list, img.shape[0]))
                wrists.append(wrist_world(world_lm))

                elbow_angle = detector.find_angle(img, R_SHOULDER, R_ELBOW, R_WRIST, draw=False)
                knee_angle = detector.find_angle(img, R_HIP, R_KNEE, R_ANKLE, draw=False)

                r_foot_angle = detector.find_vector_angle(R_HEEL, R_FOOT_INDEX)
                l_foot_angle = detector.find_vector_angle(L_HEEL, L_FOOT_INDEX)
                feet_diff = abs(abs(r_foot_angle) - abs(l_foot_angle))

                shoulder_y = lm_list[R_SHOULDER][2]
                wrist_y = lm_list[R_WRIST][2]

                registered = logic.check_shot(elbow_angle, knee_angle, feet_diff, wrist_y, shoulder_y)
                if registered:
                    logic.session_data[-1]["angles_3d"] = compute_joint_angles(world_lm)
                    logic.session_data[-1]["release_speed"] = release_speed(
                        times, wrists, len(times) - 1)

            if on_progress is not None:
                on_progress(frames_total)
    finally:
        cap.release()

    view, view_confidence = view_agg.result()
    return AnalysisResult(
        video_path=str(video_path),
        frames_total=frames_total,
        frames_with_pose=frames_with_pose,
        shots=logic.shot_count,
        session_data=logic.session_data,
        view=view,
        view_confidence=view_confidence,
        air_time=air_time(times, ankle_ys),
    )
