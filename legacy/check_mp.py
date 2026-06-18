import mediapipe as mp
import os

print(f"Путь к библиотеке: {os.path.dirname(mp.__file__)}")
try:
    print(f"Solutions доступен: {mp.solutions}")
    print("✅ ВСЁ РАБОТАЕТ! Можно запускать main.py")
except AttributeError:
    print("❌ ОШИБКА: mp.solutions всё еще не найден.")
    print("Доступные атрибуты:", dir(mp))