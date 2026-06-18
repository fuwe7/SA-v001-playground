"""Smoke-тест пайплайна анализа.

Генерирует короткое синтетическое видео (без человека) и проверяет, что
analyze_video отрабатывает без ошибок и корректно сообщает об отсутствии бросков.
Реальная проверка биомеханики делается на видео с человеком вручную.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.analyzer import analyze_video


def _make_dummy_video(path: Path, frames: int = 10, size: int = 128) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (size, size))
    for _ in range(frames):
        writer.write(np.zeros((size, size, 3), dtype=np.uint8))
    writer.release()


def test_analyze_dummy_video(tmp_path: Path) -> None:
    video = tmp_path / "dummy.mp4"
    _make_dummy_video(video)

    result = analyze_video(video)

    assert result.frames_total > 0
    assert result.frames_with_pose == 0  # человека в кадре нет
    assert result.shots == 0
    assert result.session_data == []


def test_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.mp4"
    try:
        analyze_video(missing)
    except FileNotFoundError:
        return
    raise AssertionError("ожидали FileNotFoundError для несуществующего файла")
