import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import logging
from logging.handlers import RotatingFileHandler
import os
import numpy as np

BACKEND_URL = "http://0.0.0.0:8080"

print("Starting app...")
log_filename = "app.log"

try:
    with open(f"logs/{log_filename}", 'x') as f:
        f.write("Streamlit app log file.\n")
    print(f"{log_filename} created.")
except FileExistsError:
    print(f"{log_filename} already exists.")

# Логирование с ротацией
if not os.path.exists("logs"):
    os.makedirs("logs")
log_handler = RotatingFileHandler("logs/app.log", maxBytes=1_000_000, backupCount=5)
log_handler.setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

print("App started.")

if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None

st.set_page_config(page_title="ML Model Manager", layout="wide")

st.title("ML Model Manager")

# --- 1. Загрузка датасета ---
st.header("1. Upload your dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
dataset_path = None

if uploaded_file:
    # Сохраняем локально для предпросмотра
    df = pd.read_csv(uploaded_file, sep=None, engine='python')
    st.success("Dataset loaded!")
    st.dataframe(df.head())

    # Сбрасываем позицию "курсора" в файле в начало
    uploaded_file.seek(0)

    # Загружаем на бэкенд
    files = {"file": uploaded_file}
    response = requests.post(f"{BACKEND_URL}/upload_dataset", files=files)
    if response.ok:
        dataset_path = response.json()["file_path"]
        st.success(f"Dataset uploaded to server: {dataset_path}")
    else:
        st.error("Failed to upload dataset to server")

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

    # Корреляционная матрица
    if len(df.select_dtypes(include="number").columns) > 1:
        st.subheader("Correlation Matrix")
        corr = df.select_dtypes(include="number").corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)

    logger.info(f"EDA shown for column {col}.")

# --- 3. Создание модели и выбор гиперпараметров ---
st.header("3. Create and train a model")

# Получаем доступные модели
models_resp = requests.get(f"{BACKEND_URL}/models")
models = []
if models_resp.ok:
    models = models_resp.json()

# Секция создания модели
with st.expander("Create New Model", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        new_model_name = st.text_input("Model name", value="MyModel")
    with col2:
        model_type = st.selectbox(
            "Model type",
            ["xgboost", "randomforest"],
            format_func=lambda x: "XGBoost Classifier" if x == "xgboost" else "Random Forest Classifier"
        )

    if st.button("Create Model"):
        payload = {
            "name": new_model_name,
            "model_type": model_type
        }
        resp = requests.post(f"{BACKEND_URL}/create_model", json=payload)
        if resp.ok:
            st.success(f"Model created: {resp.json()['message']}")
            # Обновляем список моделей
            models_resp = requests.get(f"{BACKEND_URL}/models")
            if models_resp.ok:
                models = models_resp.json()
        else:
            st.error("Failed to create model")

# Выбор и обучение модели
if models:
    st.subheader("Select and Train Model")

    # Выбираем модель для обучения
    model_options = {m["name"]: m for m in models}
    selected_model_name = st.selectbox("Select model to train", list(model_options.keys()))
    selected_model = model_options[selected_model_name]

    # Устанавливаем как активную, если еще не активна
    if not selected_model["is_active"]:
        if st.button("Set as Active Model"):
            resp = requests.post(f"{BACKEND_URL}/set", json={"id": selected_model["id"]})
            if resp.ok:
                st.success("Model set as active")
                # Обновлеяем модели
                models_resp = requests.get(f"{BACKEND_URL}/models")
                if models_resp.ok:
                    models = models_resp.json()
                    model_options = {m["name"]: m for m in models}
                    selected_model = next((m for m in models if m["id"] == selected_model["id"]), None)
            else:
                st.error("Failed to set model as active")

    # Отображаем гиперпараметры в зависимости от типа модели
    st.subheader(f"Configure Hyperparameters for {selected_model_name}")

    hyperparameters = {}

    if selected_model["type"] == "xgboost":
        col1, col2, col3 = st.columns(3)
        with col1:
            hyperparameters["learning_rate"] = st.slider("Learning Rate", 0.001, 1.0, 0.1)
        with col2:
            hyperparameters["n_estimators"] = st.slider("Number of Estimators", 10, 1000, 100)
        with col3:
            hyperparameters["max_depth"] = st.slider("Max Depth", 1, 15, 3)

    elif selected_model["type"] == "randomforest":
        col1, col2 = st.columns(2)
        with col1:
            hyperparameters["n_estimators"] = st.slider("Number of Estimators", 10, 500, 100)
        with col2:
            max_depth = st.slider("Max Depth (0 for None)", 0, 30, 0)
            hyperparameters["max_depth"] = None if max_depth == 0 else max_depth

    # Добавляем путь к датасету, если доступен
    if dataset_path:
        hyperparameters["data_path"] = dataset_path

    # Кнопка обучения
    if st.button("Train Model"):
        payload = {"hyperparameters": hyperparameters}
        resp = requests.post(f"{BACKEND_URL}/fit", json=payload)
        if resp.ok:
            st.success(f"Model training: {resp.json()['message']}")
            logger.info(f"Model training started with params: {payload}")
        else:
            st.error(f"Training failed: {resp.text}")
            logger.error("Training failed")

# --- 4. Просмотр информации о моделях и кривых обучения ---
st.header("4. Models and learning curves")
# Обновляем список моделей
models_resp = requests.get(f"{BACKEND_URL}/models")
if models_resp.ok:
    models = models_resp.json()
    if models:
        model_options = {m["name"]: m for m in models}
        selected_model_name = st.selectbox("Select model", list(model_options.keys()), key="view_model")
        selected_model = model_options[selected_model_name]

        # Отображаем информацию о модели
        st.subheader("Model Information")
        info_cols = st.columns(3)
        with info_cols[0]:
            st.write("**ID:**", selected_model["id"])
            st.write("**Type:**", selected_model["type"])
        with info_cols[1]:
            st.write("**Name:**", selected_model["name"])
            st.write("**Active:**", "Yes" if selected_model["is_active"] else "No")
        with info_cols[2]:
            st.write("**Description:**", selected_model["description"])

        # Отображаем метрики, если доступны
        if "metrics" in selected_model and selected_model["metrics"]:
            st.subheader("Model Metrics")
            metrics = selected_model["metrics"]

            # Отображаем метрики в зависимости от типа модели
            if selected_model["type"] == "xgboost":
                if "train_loss" in metrics and "val_loss" in metrics:
                    cols = st.columns(2)
                    with cols[0]:
                        st.metric("Training Loss", f"{metrics['train_loss']:.4f}")
                    with cols[1]:
                        st.metric("Validation Loss", f"{metrics['val_loss']:.4f}")

            elif selected_model["type"] == "randomforest":
                if "train_accuracy" in metrics and "val_accuracy" in metrics:
                    cols = st.columns(2)
                    with cols[0]:
                        st.metric("Training Accuracy", f"{metrics['train_accuracy']:.2%}")
                    with cols[1]:
                        st.metric("Validation Accuracy", f"{metrics['val_accuracy']:.2%}")

            # Отображаем график, если доступен
            if "metrics" in selected_model and selected_model["metrics"] and "plot_path" in selected_model["metrics"]:
                st.subheader("Model Visualization")

                # Пытаемся получить интерактивный график Plotly
                try:
                    plotly_resp = requests.get(f"{BACKEND_URL}/plotly_plots/{selected_model['id']}")
                    if plotly_resp.ok:
                        # Отображаем интерактивный график Plotly
                        fig_data = plotly_resp.json()

                        # Создаем новую фигуру из данных
                        if 'data' in fig_data:
                            fig = go.Figure(data=fig_data['data'], layout=fig_data.get('layout', {}))
                        else:
                            fig = go.Figure(fig_data)

                        fig.update_layout(
                            width=800,
                            height=500,
                            autosize=True
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Отображаем график картинкой, если нет данных Plotly
                        plot_path = selected_model["metrics"]["plot_path"]
                        plot_resp = requests.get(f"{BACKEND_URL}/plots/{os.path.basename(plot_path)}")
                        if plot_resp.ok:
                            st.image(plot_resp.content, caption=f"{selected_model_name} Visualization")
                        else:
                            st.warning("Plot image not available")
                except Exception as e:
                    # Отображаем график картинкой, если возникают ошибки
                    st.warning(f"Could not load interactive plot: {str(e)}")
                    plot_path = selected_model["metrics"]["plot_path"]
                    plot_resp = requests.get(f"{BACKEND_URL}/plots/{os.path.basename(plot_path)}")
                    if plot_resp.ok:
                        st.image(plot_resp.content, caption=f"{selected_model_name} Visualization")
                    else:
                        st.warning("Plot image not available")

        # Если нет реальных метрик, показываем демо-график
        else:
            st.subheader("Demo Learning Curve")
            epochs = list(range(1, 11))
            loss = [1 / (e + 0.5) for e in epochs]
            fig = px.line(x=epochs, y=loss, labels={"x": "Epoch", "y": "Loss"}, title="Demo Learning Curve")
            st.plotly_chart(fig, use_container_width=True)

        logger.info(f"Showed info and curve for model {selected_model_name}")

# --- 5. Инференс ---
st.header("5. Inference")

# Проверяем, есть ли датасет
if 'df' in locals():
    st.info(
        f"Your dataset has {df.shape[1] - 1} features. Make sure to provide the same number of values for prediction.")

# Выбор метода ввода
input_method = st.radio("Select input method", ["Manual Input", "Sample from Dataset"])

if input_method == "Manual Input":
    input_data = st.text_area("Enter input data for prediction (comma-separated values)", "1.0,2.0,3.0")
    try:
        data = [float(x) for x in input_data.split(",")]
    except:
        st.error("Invalid input format. Please enter comma-separated numbers.")
        data = None
else:
    if 'df' in locals():
        # Выбор рандомной строки из датасета
        if st.button("Get Random Sample"):
            sample_idx = np.random.randint(0, len(df))
            sample = df.iloc[sample_idx, :-1]
            st.write("Sample features:")
            st.write(sample)
            st.session_state.sample_data = sample.values.tolist()
            print(f"Sample data: {st.session_state.sample_data}")

        # Используем данные из session state
        data = st.session_state.sample_data
    else:
        st.warning("No dataset available. Please upload a dataset first.")
        data = None

if data and st.button("Predict"):
    print("Sending prediction request to backend...")
    print(f"Prediction data: {data}")
    payload = {"data": data}
    resp = requests.post(f"{BACKEND_URL}/predict", json=payload)
    print("Predict request sent to backend. Waiting for response...")
    print(f"Backend predict response: {resp.text}")
    if resp.ok:
        prediction = resp.json()['predictions']
        st.success(f"Prediction: {prediction}")
        logger.info(f"Prediction done: {prediction}")
    else:
        st.error(f"Prediction failed: {resp.text}")
        logger.error("Prediction failed")
