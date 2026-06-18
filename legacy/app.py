import streamlit as st
import cv2
import tempfile
import pandas as pd
import plotly.express as px
from core.detector import PoseDetector
from core.state_machine import ShotDetector

# --- Настройка страницы ---
st.set_page_config(page_title="HoopsLab AI", layout="wide")

st.title("🏀 HoopsLab: AI Biomechanics Analysis")
st.markdown("Загрузи видео своего броска, и ИИ разберет твою технику.")

# --- Сайдбар (Настройки) ---
st.sidebar.header("⚙️ Настройки Анализа")
perfect_thresh = st.sidebar.slider("Порог 'Идеального броска' (градусы)", 150, 180, 165)
good_thresh = st.sidebar.slider("Порог 'Хорошего броска' (градусы)", 130, 160, 150)
use_webcam = st.sidebar.checkbox("Использовать веб-камеру (вместо файла)")

# --- Загрузка видео ---
video_file = None

if not use_webcam:
    uploaded_file = st.file_uploader("Выбери видео (MP4/MOV)", type=["mp4", "mov"])
    if uploaded_file is not None:
        # Streamlit не читает файлы напрямую, сохраняем во временный файл
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_file = tfile.name
else:
    video_file = 0  # ID веб-камеры

# --- Кнопка старта ---
start_button = st.button("🚀 Запустить анализ")

if start_button and video_file is not None:
    # Контейнеры для вывода
    col1, col2 = st.columns([2, 1])

    with col1:
        st_frame = st.empty()  # Сюда будем стримить видео
    with col2:
        st_stats = st.empty()  # Сюда статистику

    # Инициализация
    cap = cv2.VideoCapture(video_file)
    detector = PoseDetector()
    logic = ShotDetector()

    # Список для хранения данных о каждом броске (для графика)
    shot_data = []

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        # 1. AI Vision
        img = detector.find_pose(img, draw=False)
        lm_list = detector.find_position(img, draw=False)

        # 2. Logic & Math
        current_angle = 0
        status = "IDLE"

        if len(lm_list) != 0:
            # Правая рука: 12-14-16
            shoulder_y = lm_list[12][2]
            wrist_y = lm_list[16][2]
            current_angle = detector.find_angle(img, 12, 14, 16, draw=True)

            # --- ВАЖНО: Модифицируем логику под настройки из слайдера ---
            # Напрямую используем метод, но перехватываем результат для графика
            # (Для простоты используем ту же логику state_machine,
            # но можно передать параметры в класс, если доработать его __init__)

            is_shot = logic.check_shot(current_angle, wrist_y, shoulder_y)

            if is_shot:
                # Оценка на основе ТВОИХ настроек слайдера
                grade = "OK"
                if current_angle >= perfect_thresh:
                    grade = "PERFECT"
                elif current_angle >= good_thresh:
                    grade = "GOOD"
                else:
                    grade = "BAD"

                # Сохраняем данные броска
                shot_data.append({
                    "Shot #": logic.shot_count,
                    "Angle": int(current_angle),
                    "Grade": grade
                })

        # 3. Визуализация на видео
        cv2.putText(img, f"Shots: {logic.shot_count}", (50, 100),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 100, 0), 3)

        # Конвертация BGR -> RGB для браузера
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st_frame.image(img, channels="RGB", use_column_width=True)

        # Обновляем статистику справа в реальном времени
        with st_stats.container():
            st.metric("Всего бросков", logic.shot_count)
            st.metric("Текущий угол", f"{int(current_angle)}°")
            if shot_data:
                last_shot = shot_data[-1]
                st.info(f"Последний: {last_shot['Grade']} ({last_shot['Angle']}°)")

    cap.release()
    st.success("Анализ завершен!")

    # --- Аналитика после матча ---
    if shot_data:
        st.divider()
        st.subheader("📊 Анализ сессии")

        # Превращаем данные в таблицу Pandas
        df = pd.DataFrame(shot_data)

        # Строим график
        fig = px.line(df, x="Shot #", y="Angle", title="Стабильность угла вылета", markers=True)

        # Добавляем линию "Идеала" (твой порог)
        fig.add_hline(y=perfect_thresh, line_dash="dash", line_color="green", annotation_text="Target")

        st.plotly_chart(fig, use_container_width=True)

        # Таблица данных
        st.dataframe(df)