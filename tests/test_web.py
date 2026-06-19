"""Smoke-тесты веб-интерфейса (Этап 1.5)."""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest

from web.app import create_app


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def _tiny_video(path: Path) -> None:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    for _ in range(3):
        w.write(np.zeros((64, 64, 3), dtype=np.uint8))
    w.release()


def test_index_ok(client) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Анализатор".encode() in resp.data


def test_analyze_rejects_non_video(client) -> None:
    data = {"video": (io.BytesIO(b"not a video"), "notes.txt")}
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_analyze_missing_file(client) -> None:
    resp = client.post("/analyze", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_analyze_processes_video(client, tmp_path) -> None:
    vid = tmp_path / "clip.mp4"
    _tiny_video(vid)
    data = {"video": (vid.open("rb"), "clip.mp4")}
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert "Отчёт".encode() in resp.data
