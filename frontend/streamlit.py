import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import logging
from logging.handlers import RotatingFileHandler
import os

BACKEND_URL = "http://backend:8000"  # или http://localhost:8000 если локально

# Логирование с ротацией
if not os.path.exists("logs"):
    os.makedirs("logs")
log_handler = RotatingFileHandler("logs/streamlit.log", maxBytes=1_000_000, backupCount=5)
log_handler.setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

st.set_page_config(page_title="ML Model Manager", layout="wide")

st.title("ML Model Manager")

# --- 1. Загрузка датасета ---
st.header("1. Upload your dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded!")
    st.dataframe(df.head())
    logger.info("Dataset uploaded and previewed.")

    # --- 2. Аналитика / EDA ---
    st.header("2. Dataset EDA")
    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", df.columns.tolist())
    st.write("**Missing values:**")
    st.write(df.isnull().sum())
    st.write("**Describe:**")
    st.write(df.describe())
    # Гистограммы
    col = st.selectbox("Select column for histogram", df.select_dtypes(include="number").columns)
    fig = px.histogram(df, x=col)
    st.plotly_chart(fig, use_container_width=True)
    logger.info(f"EDA shown for column {col}.")

    # --- 3. Создание модели и выбор гиперпараметров ---
    st.header("3. Create and train a model")
    model_name = st.text_input("Model name", value="MyModel")
    hp1 = st.slider("Hyperparameter 1 (learning_rate)", 0.001, 1.0, 0.01)
    hp2 = st.number_input("Hyperparameter 2 (n_estimators)", 10, 1000, 100)
    if st.button("Train model"):
        # Пример отправки на backend
        payload = {
            "hyperparameters": {"learning_rate": hp1, "n_estimators": hp2, "sleep": 2}
        }
        resp = requests.post(f"{BACKEND_URL}/fit", json=payload)
        if resp.ok:
            st.success(f"Model training started: {resp.json()['message']}")
            logger.info(f"Model training started with params: {payload}")
        else:
            st.error("Training failed")
            logger.error("Training failed")

# --- 4. Просмотр информации о моделях и кривых обучения ---
st.header("4. Models and learning curves")
models_resp = requests.get(f"{BACKEND_URL}/models")
if models_resp.ok:
    models = models_resp.json()
    if models:
        model_options = {m["name"]: m["id"] for m in models}
        selected_model = st.selectbox("Select model", list(model_options.keys()))
        model_id = model_options[selected_model]
        info = next(m for m in models if m["id"] == model_id)
        st.json(info)
        # Заглушка: кривая обучения (демо)
        epochs = list(range(1, 11))
        loss = [1 / (e + 0.5) for e in epochs]
        fig = px.line(x=epochs, y=loss, labels={"x": "Epoch", "y": "Loss"}, title="Demo Learning Curve")
        st.plotly_chart(fig, use_container_width=True)
        logger.info(f"Showed info and curve for model {selected_model}")

# --- 5. Инференс ---
st.header("5. Inference")
input_data = st.text_area("Enter input data for prediction (comma-separated values)", "1.0,2.0,3.0")
if st.button("Predict"):
    try:
        data = [float(x) for x in input_data.split(",")]
        payload = {"data": data}
        resp = requests.post(f"{BACKEND_URL}/predict", json=payload)
        if resp.ok:
            st.success(f"Prediction: {resp.json()['predictions']}")
            logger.info(f"Prediction done: {resp.json()['predictions']}")
        else:
            st.error("Prediction failed")
            logger.error("Prediction failed")
    except Exception as e:
        st.error(f"Invalid input: {e}")

# --- Бонус: сравнение экспериментов ---
st.header("Bonus: Compare experiments")
if models_resp.ok and len(models) > 1:
    selected = st.multiselect("Select models to compare (up to 5)", list(model_options.keys()))
    if selected:
        fig = px.line()
        for name in selected[:5]:
            epochs = list(range(1, 11))
            loss = [1 / (e + 0.5 + i) for i, e in enumerate(epochs)]  # демо-данные
            fig.add_scatter(x=epochs, y=loss, mode='lines', name=name)
        fig.update_layout(title="Learning Curves Comparison", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig, use_container_width=True)
        logger.info(f"Compared experiments: {selected}")

st.info("All logs are stored in the logs/ folder (with rotation).")
