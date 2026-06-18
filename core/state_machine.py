import time


class ShotDetector:
    def __init__(self):
        self.shot_count = 0
        self.state = "IDLE"
        self.feedback = ""

        # Хранилище всех бросков сессии
        # Структура: [{'id': 1, 'elbow': 170, 'knee': 130, ...}, ...]
        self.session_data = []

        # Временные переменные для текущего броска
        self.temp_min_knee = 180  # Ищем минимальный угол (глубину приседа)
        self.temp_set_point = 0  # Угол локтя в начале фазы
        self.temp_feet_diff = 0

    def check_shot(self, elbow_angle, knee_angle, feet_diff, wrist_y, shoulder_y):
        """
        Теперь принимает knee_angle и feet_diff для сбора статистики
        """

        # 1. СБРОС (IDLE)
        if wrist_y > shoulder_y:
            if self.state != "IDLE":
                # Сбрасываем временные метрики при переходе в IDLE
                self.state = "IDLE"
                self.feedback = ""
                self.temp_min_knee = 180
            return False

        # 2. ПОДГОТОВКА (PREP)
        if self.state == "IDLE" and wrist_y < shoulder_y:
            if elbow_angle < 110:
                self.state = "PREP"
                self.feedback = "Load..."
                self.temp_set_point = elbow_angle  # Запоминаем точку выноса

        # В фазе PREP мы постоянно ищем САМЫЙ глубокий присед
        if self.state == "PREP":
            if knee_angle < self.temp_min_knee:
                self.temp_min_knee = knee_angle

            # Запоминаем текущее положение ног
            self.temp_feet_diff = feet_diff

        # 3. БРОСОК (RELEASE)
        if self.state == "PREP":
            # Если рука выпрямилась
            if elbow_angle > 140:
                self.state = "RELEASE"
                self.shot_count += 1

                # --- АНАЛИЗ КАЧЕСТВА ---
                grade = "OK"
                if elbow_angle > 160: grade = "Great Arc"
                if elbow_angle < 145: grade = "Short Arm"
                if self.temp_min_knee > 150: grade += " (No Legs)"

                self.feedback = grade

                # --- СОХРАНЕНИЕ СТАТИСТИКИ ---
                shot_record = {
                    "id": self.shot_count,
                    "elbow_release": int(elbow_angle),
                    "knee_load": int(self.temp_min_knee),  # Самая глубокая точка
                    "set_point": int(self.temp_set_point),
                    "feet_diff": int(self.temp_feet_diff),
                    "grade": grade,
                    "timestamp": time.strftime("%H:%M:%S")
                }
                self.session_data.append(shot_record)

                print(f"Shot {self.shot_count}: Elbow {int(elbow_angle)} | Knee Load {int(self.temp_min_knee)}")
                return True

        return False