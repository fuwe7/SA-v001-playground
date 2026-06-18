from ultralytics import YOLO
import numpy as np
from collections import deque


class BallTracker:
    def __init__(self, model_path='yolov8s.pt', buffer_size=15):  # Хвост покороче (15), чтобы не мешал
        # Повышаем уверенность до 0.35, чтобы убрать мусор
        self.model = YOLO(model_path)
        self.trajectory = deque(maxlen=buffer_size)
        self.last_pos = None
        self.rim_box = None

    def detect_ball(self, img):
        # 1. Поиск (conf=0.35 - фильтруем шум)
        results = self.model(img, stream=True, verbose=False, conf=0.35, device=0)

        best_pos = None
        min_dist = 99999  # Ищем мяч, который БЛИЖЕ ВСЕГО к предыдущему

        candidates = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                if int(box.cls[0]) == 32:  # Sports ball
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    candidates.append((cx, cy))

        # 2. Логика отсеивания "Телепортации"
        if not candidates:
            return None

        if self.last_pos is None:
            # Если мяча не было, берем любой первый (начало броска)
            best_pos = candidates[0]
        else:
            # Если мяч был, ищем ближайшего кандидата
            for center in candidates:
                # Считаем расстояние от прошлого кадра (Пифагор)
                dist = np.linalg.norm(np.array(center) - np.array(self.last_pos))

                # ВАЖНО: Если мяч "улетел" больше чем на 150 пикселей за кадр - это баг
                if dist < 150 and dist < min_dist:
                    min_dist = dist
                    best_pos = center

        if best_pos:
            self.last_pos = best_pos

        return best_pos

    def update_trajectory(self, center):
        if center:
            self.trajectory.append(center)
        else:
            # Если мяч потерян - НЕ добавляем None, просто ничего не делаем,
            # либо очищаем хвост, если потерян давно (опционально)
            pass

        return self.trajectory

    def check_score(self, rim_rect):
        """
        Простая логика попадания: Вектор вниз + Внутри коробки
        """
        if len(self.trajectory) < 3 or rim_rect is None:
            return False

        # Берем 3 последние точки для надежности
        current = self.trajectory[-1]
        prev = self.trajectory[-3]  # Смотрим чуть назад

        rx, ry, rw, rh = rim_rect

        # 1. Мяч внутри ширины кольца?
        if not (rx < current[0] < rx + rw):
            return False

        # 2. Мяч внутри высоты кольца?
        if not (ry < current[1] < ry + rh):
            return False

        # 3. Мяч движется ВНИЗ? (y растет)
        if current[1] > prev[1] + 10:  # +10 пикселей вниз минимум
            return True

        return False