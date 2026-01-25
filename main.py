import cv2
from core.detector import PoseDetector
from core.state_machine import ShotDetector  # Импортируем нашу логику


def main():
    video_file = 'test_shot1.mp4'
    cap = cv2.VideoCapture(video_file)

    detector = PoseDetector()
    shot_logic = ShotDetector()  # Инициализируем "мозги"

    while True:
        success, img = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # 1. Получаем данные CV
        img = detector.find_pose(img, draw=False)
        lm_list = detector.find_position(img, draw=False)

        if len(lm_list) != 0:
            # Координаты для правой руки:
            # 12 - Плечо, 14 - Локоть, 16 - Запястье

            # ВАЖНО: lm_list[12][2] - это Y координата плеча ([id, x, y])
            shoulder_y = lm_list[12][2]
            wrist_y = lm_list[16][2]

            # Считаем угол
            angle = detector.find_angle(img, 12, 14, 16)

            # 2. Передаем данные в логику
            is_shot = shot_logic.check_shot(angle, wrist_y, shoulder_y)

            # 3. Визуализация (UI)
            # Выводим счетчик бросков
            cv2.putText(img, f"Shots: {shot_logic.shot_count}", (50, 100),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # Выводим текущее состояние (IDLE/PREP/RELEASE)
            cv2.putText(img, f"State: {shot_logic.state}", (50, 150),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            # Если есть фидбек (Good/Bad) - пишем его крупно
            if shot_logic.feedback:
                color = (0, 255, 0) if "PERFECT" in shot_logic.feedback else (0, 165, 255)
                cv2.putText(img, shot_logic.feedback, (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.imshow("HoopsLab v0.3", img)

        if cv2.waitKey(20) & 0xFF == ord('q'):  # Чуть замедлил (20мс), чтобы успевать видеть
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()