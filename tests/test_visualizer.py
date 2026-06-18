"""Smoke-тест визуализации (headless-режим: запись в файл, без окон)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.visualizer import render_stats_panel, run_visualization


def _make_dummy_video(path: Path, frames: int = 8, size: int = 128) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (size, size))
    for _ in range(frames):
        writer.write(np.zeros((size, size, 3), dtype=np.uint8))
    writer.release()


def test_save_video_runs(tmp_path: Path) -> None:
    src = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    _make_dummy_video(src)

    logic = run_visualization(src, show=False, save_path=str(out))

    assert out.exists() and out.stat().st_size > 0
    assert logic.shot_count == 0  # человека в кадре нет


def test_render_stats_panel_shape() -> None:
    panel = render_stats_panel({"STATUS": "IDLE", "SHOTS": 0}, height=480)
    assert panel.shape[0] == 480
    assert panel.shape[2] == 3
