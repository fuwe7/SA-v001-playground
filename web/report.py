"""Сборка отчёта по видео для веб-интерфейса (Этап 1.5).

Запускает headless-анализ (метрики) и рендер аннотированного видео (скелет + панель),
плюс перекодирует его в браузер-совместимый H.264, если доступен ffmpeg.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from core.analyzer import AnalysisResult, analyze_video
from core.view import VIEW_ANGLED, VIEW_FRONT, VIEW_SIDE
from core.visualizer import run_visualization

VIEW_HINT = {
    VIEW_SIDE: "Снято сбоку — надёжны углы локтя, колена и бедра.",
    VIEW_FRONT: "Снято спереди/сзади — надёжны симметрия плеч и постановка стоп; "
                "углы сгиба менее точны.",
    VIEW_ANGLED: "Снято наискосок — для точных углов снимайте строго сбоку.",
}


@dataclass
class Report:
    result: AnalysisResult
    video_name: str | None  # имя файла превью-видео в папке отчёта (или None)
    view_hint: str
    shots: list = field(default_factory=list)


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _transcode_h264(src: Path, dst: Path) -> bool:
    """Перекодирует видео в H.264 (mp4) для воспроизведения в браузере."""
    if not _has_ffmpeg():
        return False
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an", str(dst),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc.returncode == 0 and dst.exists()


def build_report(video_path: str | Path, out_dir: str | Path) -> Report:
    """Анализирует видео и готовит ассеты отчёта в out_dir.

    Returns:
        Report с метриками и именем файла аннотированного видео (если удалось собрать).
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = analyze_video(video_path)

    raw = out_dir / "annotated_raw.mp4"
    run_visualization(video_path, show=False, save_path=str(raw))

    video_name: str | None = None
    if raw.exists():
        web = out_dir / "annotated.mp4"
        if _transcode_h264(raw, web):
            raw.unlink(missing_ok=True)
            video_name = web.name
        else:
            video_name = raw.name

    return Report(
        result=result,
        video_name=video_name,
        view_hint=VIEW_HINT.get(result.view, ""),
        shots=result.session_data,
    )


def probe_playable(path: str | Path) -> bool:
    """Проверяет, что OpenCV смог открыть видео (валидный формат)."""
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened()
    cap.release()
    return ok
