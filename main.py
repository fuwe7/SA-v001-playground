import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from core.detector import PoseDetector
from core.state_machine import ShotDetector


# --- 1. ОТРИСОВКА ПОЛНОГО СКЕЛЕТА ---
def draw_full_skeleton(img, lm_list):
    """
    Рисует ОБЕ стороны тела (левую и правую).
    """
    if len(lm_list) == 0: return

    # Все связи (Кости)
    skeleton_links = [
        # Туловище
        (11, 12), (11, 23), (12, 24), (23, 24),
        # Правая рука
        (12, 14), (14, 16),
        # Левая рука (ДОБАВИЛИ)
        (11, 13), (13, 15),
        # Правая нога
        (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
        # Левая нога (ДОБАВИЛИ)
        (23, 25), (25, 27), (27, 29), (29, 31), (27, 31)
    ]

    for p1, p2 in skeleton_links:
        if p1 < len(lm_list) and p2 < len(lm_list):
            x1, y1 = lm_list[p1][1], lm_list[p1][2]
            x2, y2 = lm_list[p2][1], lm_list[p2][2]
            # Белые тонкие линии
            cv2.line(img, (x1, y1), (x2, y2), (240, 240, 240), 1, cv2.LINE_AA)

    # Все суставы
    # 11-16 (руки), 23-32 (ноги)
    for id in range(11, 33):
        if id < len(lm_list):
            cx, cy = lm_list[id][1], lm_list[id][2]
            # Правая сторона - голубым, Левая - красным (для наглядности)
            color = (255, 100, 0) if id % 2 == 0 else (0, 100, 255)
            cv2.circle(img, (cx, cy), 3, color, -1)


# --- 2. ГЕНЕРАТОР ТАБЛИЦЫ СТАТИСТИКИ ---
def create_stats_image(metrics, width=400, height=600):
    board = np.zeros((height, width, 3), dtype=np.uint8)

    # Шапка
    cv2.rectangle(board, (0, 0), (width, 80), (30, 30, 30), -1)
    cv2.putText(board, "LIVE TELEMETRY", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    y = 120
    for key, value in metrics.items():
        # Название
        cv2.putText(board, key, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Значение и цвет
        val_str = str(value)
        color = (255, 255, 255)  # Белый по дефолту

        if "Good" in val_str or "Perfect" in val_str:
            color = (0, 255, 0)
        elif "Bad" in val_str or "Short" in val_str:
            color = (0, 0, 255)
        elif "PREP" in val_str:
            color = (0, 255, 255)
        elif "RELEASE" in val_str:
            color = (0, 255, 0)

        cv2.putText(board, val_str, (20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Линия разделитель
        y += 75
        cv2.line(board, (20, y - 20), (width - 20, y - 20), (50, 50, 50), 1)

    return board


# --- 3. ФУНКЦИЯ ОТЧЕТА ПОСЛЕ МАТЧА ---
def show_session_report(session_data):
    if not session_data:
        print("Нет данных для отчета.")
        return

    df = pd.DataFrame(session_data)
    print("\n=== RAW DATA ===")
    print(df[['id', 'elbow_release', 'knee_load', 'feet_diff', 'grade']].head())

    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    # График 1: Локоть
    axes[0].plot(df['id'], df['elbow_release'], color='#00ff00', marker='o')
    axes[0].axhline(y=160, color='white', linestyle='--', alpha=0.3)
    axes[0].set_title('Elbow Extension (Target: >160°)')
    axes[0].set_ylabel('Degrees')

    # График 2: Колени
    axes[1].plot(df['id'], df['knee_load'], color='#00aaff', marker='s')
    axes[1].axhline(y=140, color='white', linestyle='--', alpha=0.3)
    axes[1].set_title('Knee Load Depth (Lower is Better)')
    axes[1].set_ylabel('Degrees')

    # График 3: Стопы
    colors = ['green' if x < 20 else 'red' for x in df['feet_diff']]
    axes[2].bar(df['id'], df['feet_diff'], color=colors)
    axes[2].set_title('Feet Parallelism Diff (Lower is Better)')
    axes[2].set_xlabel('Shot #')

    plt.tight_layout()
    plt.show()


# --- MAIN ---
def main():
    video_file = 'test_shot.mp4'
    cap = cv2.VideoCapture(video_file)

    detector = PoseDetector()
    shot_logic = ShotDetector()

    print(">>> Анализ запущен. Нажми 'Q' для выхода.")

    while True:
        success, img = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Включаем Z-координаты для проверки перспективы
        # Но для find_pose нам нужны пиксели
        img = detector.find_pose(img, draw=False)
        lm_list = detector.find_position(img, draw=False)

        # Получаем "Сырые" landmarks (с координатами Z) для проверки глубины
        # results.pose_landmarks хранит нормализованные x, y, z (от -1 до 1)
        raw_landmarks = detector.results.pose_landmarks

        metrics = {
            "STATUS": "Waiting",
            "SHOTS": 0,
            "FEET DIFF": "--",
            "ELBOW": "--",
            "KNEE": "--"
        }

        if len(lm_list) != 0:
            # 1. Рисуем полный скелет
            draw_full_skeleton(img, lm_list)

            # 2. Считаем метрики
            elbow_angle = detector.find_angle(img, 12, 14, 16, draw=False)
            knee_angle = detector.find_angle(img, 24, 26, 28, draw=False)

            # 3. Анализ стоп (Векторы)
            r_foot_angle = detector.find_vector_angle(30, 32)
            l_foot_angle = detector.find_vector_angle(29, 31)
            feet_diff = abs(abs(r_foot_angle) - abs(l_foot_angle))

            # --- ПРОВЕРКА РАКУРСА (Z-Check) ---
            # Сравниваем Z (глубину) левой пятки (29) и правой пятки (30)
            # Если разница большая - игрок стоит боком
            z_left = raw_landmarks.landmark[29].z
            z_right = raw_landmarks.landmark[30].z
            z_diff = abs(z_left - z_right)

            feet_status = f"{int(feet_diff)} deg"

            # Если Z-diff большой (> 0.1 условно), значит ракурс плохой
            if z_diff > 0.15:
                feet_status += " (Bad Cam Angle)"
            elif feet_diff < 15:
                feet_status += " (Good)"
            else:
                feet_status += " (Wide)"

            # 4. Логика броска
            shoulder_y = lm_list[12][2]
            wrist_y = lm_list[16][2]
            shot_logic.check_shot(elbow_angle, knee_angle, feet_diff, wrist_y, shoulder_y)

            # 5. Обновляем метрики
            metrics["STATUS"] = shot_logic.state
            if shot_logic.feedback:
                metrics["STATUS"] = shot_logic.feedback  # Показываем фидбек (Great Arc и тд)

            metrics["SHOTS"] = shot_logic.shot_count
            metrics["FEET DIFF"] = feet_status
            metrics["ELBOW"] = f"{int(elbow_angle)} deg"
            metrics["KNEE"] = f"{int(knee_angle)} deg"

        # Отрисовка двух окон
        stats_img = create_stats_image(metrics)
        cv2.imshow("Video Feed", img)
        cv2.imshow("Live Stats", stats_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    show_session_report(shot_logic.session_data)


if __name__ == "__main__":
    main()