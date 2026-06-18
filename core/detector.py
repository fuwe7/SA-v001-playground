import cv2
import mediapipe as mp
import math


class PoseDetector:
    # Добавь этот метод внутрь класса PoseDetector в core/detector.py
    def find_vector_angle(self, p1, p2):
        """
        Считает угол наклона прямой между точками p1 и p2 относительно горизонта.
        0 градусов - горизонтальная линия.
        90 градусов - вертикальная.
        """
        if len(self.lm_list) < p2:
            return 0

        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]

        # Считаем угол вектора
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        return angle

    def __init__(self, mode=False, complexity=1, smooth=True,
                 detection_con=0.5, track_con=0.5):

        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.detection_con = detection_con
        self.track_con = track_con

        # Стандартный импорт
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.complexity,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.lm_list = []



    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(
                    img,
                    self.results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
        return img

    def find_position(self, img, draw=True):
        self.lm_list = []
        if self.results.pose_landmarks:
            h, w, c = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lm_list

    def find_angle(self, img, p1, p2, p3, draw=True):
        """
        Считает угол между точками p1, p2, p3.
        Исправлена логика для диапазона 0-180 градусов.
        """
        # Получаем координаты
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        x3, y3 = self.lm_list[p3][1:]

        # Считаем угол
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))

        # 1. Приводим отрицательные к положительным
        if angle < 0:
            angle += 360

        # 2. Берем меньший угол (внутренний).
        # Если получилось 270 (внешний), то внутренний = 360 - 270 = 90.
        if angle > 180:
            angle = 360 - angle

        if draw:
            # Рисуем линии
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)

            # Рисуем суставы
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)

            # Пишем текст (чуть крупнее и зеленым)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        return angle