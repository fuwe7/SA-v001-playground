"""Flask-приложение «видео → отчёт» (Этап 1.5).

Запуск:
    python -m web.app            # сервер на http://127.0.0.1:5000
"""
from __future__ import annotations

import uuid
from pathlib import Path

from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from web.report import build_report, probe_playable

ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 МБ
RESULTS_DIR = Path(__file__).resolve().parent / "_results"


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/analyze")
    def analyze():
        file = request.files.get("video")
        if file is None or file.filename == "":
            return render_template("index.html", error="Выберите видеофайл."), 400

        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            return render_template(
                "index.html",
                error=f"Формат {ext or '—'} не поддерживается. Загрузите видео (mp4/mov/...).",
            ), 400

        token = uuid.uuid4().hex
        job_dir = RESULTS_DIR / token
        job_dir.mkdir(parents=True, exist_ok=True)
        src = job_dir / f"input{ext}"
        file.save(src)

        if not probe_playable(src):
            return render_template(
                "index.html", error="Не удалось прочитать видео — файл повреждён или не видео."
            ), 400

        report = build_report(src, job_dir)
        video_url = (
            url_for("result_asset", token=token, name=report.video_name)
            if report.video_name
            else None
        )
        return render_template(
            "report.html",
            r=report,
            original=secure_filename(file.filename),
            video_url=video_url,
        )

    @app.get("/results/<token>/<name>")
    def result_asset(token: str, name: str):
        job_dir = RESULTS_DIR / secure_filename(token)
        if not job_dir.is_dir():
            abort(404)
        return send_from_directory(job_dir, secure_filename(name))

    @app.errorhandler(413)
    def too_large(_e):
        return render_template(
            "index.html", error="Файл слишком большой (лимит 200 МБ)."
        ), 413

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
