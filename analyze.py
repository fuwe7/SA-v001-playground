"""CLI-вход анализатора броска (Этап 0).

Пример:
    python analyze.py path/to/shot.mp4
    python analyze.py path/to/shot.mp4 --json result.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.analyzer import analyze_video


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Анализатор баскетбольного броска — Этап 0 (CLI, без GUI)."
    )
    parser.add_argument("video", help="путь к видеофайлу (mp4/mov)")
    parser.add_argument(
        "--json",
        dest="json_out",
        default=None,
        help="сохранить результат в JSON-файл",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="показать видео со скелетом и окно статистики (live-окна, нужен GUI)",
    )
    parser.add_argument(
        "--save-video",
        dest="save_video",
        default=None,
        help="сохранить аннотированное видео (скелет + панель статистики) в файл",
    )
    args = parser.parse_args(argv)

    video = Path(args.video)
    if not video.exists():
        print(f"Файл не найден: {video}", file=sys.stderr)
        return 1

    if args.show or args.save_video:
        from core.visualizer import run_visualization

        if args.save_video:
            print(f">>> Визуализация -> {args.save_video}")
        if args.show:
            print(">>> Открываю окна (нажми 'q' в окне видео, чтобы выйти) ...")
        logic = run_visualization(video, show=args.show, save_path=args.save_video)
        print(f"Найдено бросков: {logic.shot_count}")
        return 0

    print(f">>> Анализирую {video} ...")
    result = analyze_video(video)

    print(f"Кадров всего: {result.frames_total}")
    print(f"Кадров с распознанной позой: {result.frames_with_pose}")
    print(f"Найдено бросков: {result.shots}")

    if result.session_data:
        print("\n#  | elbow | knee_load | feet_diff | grade")
        print("-" * 50)
        for s in result.session_data:
            print(
                f"{s['id']:<2} | {s['elbow_release']:<5} | "
                f"{s['knee_load']:<9} | {s['feet_diff']:<9} | {s['grade']}"
            )
            angles = s.get("angles_3d")
            if angles:
                print("     3D-углы (град):")
                for name, value in angles.items():
                    shown = "н/д" if value is None else value
                    print(f"       {name:<16}: {shown}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON сохранён: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
