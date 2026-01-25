class ShotDetector:
    def __init__(self):
        self.shot_count = 0
        self.state = "IDLE"  # Возможные: IDLE, PREP, RELEASE
        self.feedback = ""  # Сообщение на экране (Good/Bad)

    def check_shot(self, angle, wrist_y, shoulder_y):
        """
        Принимает:
        - angle: текущий угол локтя
        - wrist_y: координата Y запястья
        - shoulder_y: координата Y плеча
        """

        # В OpenCV ось Y перевернута! 0 - это верх экрана.
        # Поэтому "Запястье выше плеча" значит wrist_y < shoulder_y

        # 1. ФАЗА IDLE (Сброс)
        # Если рука опустилась ниже плеча — мы готовы к новому броску
        if wrist_y > shoulder_y:
            self.state = "IDLE"
            self.feedback = ""
            return False

        # 2. ФАЗА PREP (Подготовка)
        # Рука поднята (wrist_y < shoulder_y), но локоть еще согнут
        if self.state == "IDLE" and wrist_y < shoulder_y:
            if angle < 110:  # Угол "заряженной" руки
                self.state = "PREP"
                self.feedback = "Aiming..."

        # 3. ФАЗА RELEASE (Бросок)
        # Мы были в PREP и выпрямили руку
        if self.state == "PREP":
            if angle > 140:  # Порог фиксации броска
                self.state = "RELEASE"
                self.shot_count += 1

                # Анализ качества (простая логика для начала)
                if angle > 165:
                    self.feedback = "PERFECT FORM!"
                elif angle > 150:
                    self.feedback = "Good Shot"
                else:
                    self.feedback = "Extend more!"

                print(f"Shot #{self.shot_count} | Angle: {int(angle)} | {self.feedback}")
                return True  # Бросок засчитан

        return False