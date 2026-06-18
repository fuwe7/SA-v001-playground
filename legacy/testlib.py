import sys

print(f"Python: {sys.version}")

try:
    import mediapipe as mp

    print(f"MediaPipe path: {mp.__file__}")

    # Момент истины
    print(f"Testing solutions... {mp.solutions}")
    print("✅ УСПЕХ! Библиотека работает. Можно запускать main.py")
except Exception as e:
    print(f"❌ ОШИБКА: {e}")
    # Выведем, что вообще есть внутри
    import mediapipe

    print(dir(mediapipe))