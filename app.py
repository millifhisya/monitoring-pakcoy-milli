
# ======================================================================
# 🌱 MONITORING PAKCOY MILLI
# ======================================================================
# FINAL HOSTING VERSION
#
# 1. Dashboard Autoplay
# 2. LSTM Prediction
# 3. CNN Classification
# 4. Analytics / History
#
# NO TRAINING
# ======================================================================

import os
import tempfile

import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PIL import Image

import joblib
import tensorflow as tf


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "Dataset_Pakcoy.xlsx"
)

LSTM_MODEL_PATH = os.path.join(
    BASE_DIR,
    "Modified_LSTM_Seq2Seq_7Days_FINAL.keras"
)

LSTM_SCALER_PATH = os.path.join(
    BASE_DIR,
    "scaler_y_pakcoy_FINAL.pkl"
)

CNN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "CNN_Modified_Spatial_Attention_Pakcoy_FINAL.keras"
)

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Fase_Vegetatif",
    "Fase_Pembentukan_Tajuk",
    "Fase_Siap_Panen"
]


# ======================================================================
# CNN CUSTOM LAYERS
# ======================================================================

@tf.keras.utils.register_keras_serializable(
    package="Pakcoy"
)
class ChannelAveragePool(tf.keras.layers.Layer):

    def call(self, inputs):

        return tf.reduce_mean(
            inputs,
            axis=-1,
            keepdims=True
        )


@tf.keras.utils.register_keras_serializable(
    package="Pakcoy"
)
class ChannelMaxPool(tf.keras.layers.Layer):

    def call(self, inputs):

        return tf.reduce_max(
            inputs,
            axis=-1,
            keepdims=True
        )


# ======================================================================
# LOAD DATASET
# ======================================================================

df_dashboard = pd.read_excel(
    DATASET_PATH
)

df_dashboard["_Time_Order"] = (
    df_dashboard["Time"]
    .astype(str)
    .map({
        "08:00": 1,
        "12:00": 2,
        "16:00": 3
    })
)

df_dashboard = (
    df_dashboard
    .sort_values(
        ["Day", "DAP", "_Time_Order"]
    )
    .reset_index(drop=True)
)

TOTAL_DATA = len(df_dashboard)

DAP_MIN = int(
    df_dashboard["DAP"].min()
)

DAP_MAX = int(
    df_dashboard["DAP"].max()
)


# ======================================================================
# LOAD LSTM
# ======================================================================

lstm_model = tf.keras.models.load_model(
    LSTM_MODEL_PATH,
    compile=False
)

lstm_scaler = joblib.load(
    LSTM_SCALER_PATH
)


# ======================================================================
# LOAD CNN
# ======================================================================

cnn_model = tf.keras.models.load_model(
    CNN_MODEL_PATH,
    compile=False,
    safe_mode=True
)


# ======================================================================
# DASHBOARD
# ======================================================================

WINDOW_SIZE = 30


def dashboard_display(index):

    index = int(index)

    index = max(
        0,
        min(index, TOTAL_DATA - 1)
    )

    row = df_dashboard.iloc[index]

    day = row["Day"]
    dap = row["DAP"]
    waktu = row["Time"]

    moisture = float(
        row["Soil Moisture (%)"]
    )

    temperature = float(
        row["Temperature (°C)"]
    )

    maturity = float(
        row["Ground Truth Maturity Level (%)"]
    )

    if maturity < 70:
        fase = "Fase Vegetatif"
    elif maturity < 90:
        fase = "Fase Pembentukan Tajuk"
    else:
        fase = "Fase Siap Panen"

    start = max(
        0,
        index - WINDOW_SIZE + 1
    )

    data = df_dashboard.iloc[
        start:index + 1
    ]

    x = np.arange(
        start + 1,
        index + 2
    )

    info = f"""
## 🌱 MONITORING PAKCOY

### Data {index + 1} / {TOTAL_DATA}

| Parameter | Nilai |
|---|---:|
| **Day** | {day} |
| **DAP / HST** | {dap} |
| **Time** | {waktu} |
| **Soil Moisture** | {moisture:.2f} % |
| **Temperature** | {temperature:.2f} °C |
| **Maturity** | {maturity:.2f} % |
| **Fase** | **{fase}** |
"""

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=[
            "Soil Moisture",
            "Temperature",
            "Ground Truth Maturity"
        ]
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=data["Soil Moisture (%)"],
            mode="lines+markers"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=data["Temperature (°C)"],
            mode="lines+markers"
        ),
        row=2,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=data[
                "Ground Truth Maturity Level (%)"
            ],
            mode="lines+markers"
        ),
        row=3,
        col=1
    )

    fig.update_yaxes(
        title_text="Moisture (%)",
        row=1,
        col=1
    )

    fig.update_yaxes(
        title_text="Temperature (°C)",
        row=2,
        col=1
    )

    fig.update_yaxes(
        title_text="Maturity (%)",
        row=3,
        col=1
    )

    fig.update_xaxes(
        title_text="Urutan Data",
        row=3,
        col=1
    )

    fig.update_layout(
        height=750,
        margin=dict(
            l=60,
            r=30,
            t=70,
            b=50
        ),
        showlegend=False
    )

    return info, fig


def dashboard_next(index):

    index = int(index)

    index += 1

    if index >= TOTAL_DATA:
        index = 0

    return index


# ======================================================================
# LSTM
# ======================================================================

def lstm_prediction(dap_start):

    try:

        dap_start = int(
            float(dap_start)
        )

        # Cari baris pertama dengan DAP tersebut
        matching = df_dashboard[
            df_dashboard["DAP"] == dap_start
        ]

        if len(matching) == 0:

            return (
                f"❌ DAP {dap_start} tidak tersedia.",
                pd.DataFrame(),
                None
            )

        start_index = matching.index[0]

        # 21 observasi input
        input_data = df_dashboard.loc[
            start_index:start_index + 20,
            [
                "DAP",
                "Soil Moisture (%)",
                "Temperature (°C)"
            ]
        ].values.astype(
            np.float32
        )

        if len(input_data) != 21:

            return (
                "❌ Data input 21 observasi tidak mencukupi.",
                pd.DataFrame(),
                None
            )

        X = np.expand_dims(
            input_data,
            axis=0
        )

        prediction_scaled = lstm_model.predict(
            X,
            verbose=0
        )

        prediction_scaled = np.asarray(
            prediction_scaled
        ).reshape(-1, 1)

        prediction = lstm_scaler.inverse_transform(
            prediction_scaled
        ).reshape(-1)

        # Future 21 observations
        future = df_dashboard.iloc[
            start_index + 21:
            start_index + 42
        ].copy()

        n = min(
            len(future),
            len(prediction)
        )

        prediction = prediction[:n]
        future = future.iloc[:n].copy()

        result = pd.DataFrame({

            "Horizon": [
                f"H{i+1}"
                for i in range(n)
            ],

            "DAP": future["DAP"].values,

            "Time": future["Time"].values,

            "Predicted Maturity (%)":
                np.round(
                    prediction,
                    2
                )
        })

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=result["DAP"],
                y=result[
                    "Predicted Maturity (%)"
                ],
                mode="lines+markers",
                name="Predicted"
            )
        )

        fig.update_layout(
            title="Prediksi Maturity Pakcoy 7 Hari",
            xaxis_title="DAP / HST",
            yaxis_title="Maturity (%)",
            height=450
        )

        summary = f"""
## 🔮 HASIL PREDIKSI LSTM

**Input DAP:** {dap_start}

**Jumlah prediksi:** {n} observasi

**Prediksi minimum:** {prediction.min():.2f} %

**Prediksi maksimum:** {prediction.max():.2f} %
"""

        return (
            summary,
            result,
            fig
        )

    except Exception as e:

        return (
            f"❌ Error LSTM: {str(e)}",
            pd.DataFrame(),
            None
        )


# ======================================================================
# CNN
# ======================================================================

def cnn_prediction(image):

    if image is None:

        return (
            "Silakan upload gambar Pakcoy.",
            None,
            None,
            pd.DataFrame()
        )

    try:

        if isinstance(image, np.ndarray):

            img = Image.fromarray(
                image.astype(np.uint8)
            )

        elif isinstance(image, Image.Image):

            img = image

        else:

            img = Image.open(
                image
            )

        img = img.convert("RGB")

        display_image = img.copy()

        img = img.resize(
            IMG_SIZE
        )

        arr = np.asarray(
            img,
            dtype=np.float32
        ) / 255.0

        batch = np.expand_dims(
            arr,
            axis=0
        )

        probabilities = cnn_model.predict(
            batch,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = float(
            probabilities[
                predicted_index
            ] * 100
        )

        table = pd.DataFrame({

            "Kelas": CLASS_NAMES,

            "Probability (%)": np.round(
                probabilities * 100,
                2
            )
        })

        fig = go.Figure(
            data=[
                go.Bar(
                    x=CLASS_NAMES,
                    y=probabilities * 100
                )
            ]
        )

        fig.update_layout(
            title="Probabilitas Klasifikasi CNN",
            xaxis_title="Kelas",
            yaxis_title="Probability (%)",
            yaxis=dict(
                range=[0, 100]
            ),
            height=400
        )

        summary = f"""
## 🖼️ HASIL KLASIFIKASI CNN

### Prediksi Utama

# {predicted_class}

### Confidence

# {confidence:.2f} %

Model menggunakan 3 kelas:

- Fase Vegetatif
- Fase Pembentukan Tajuk
- Fase Siap Panen
"""

        return (
            summary,
            display_image,
            fig,
            table
        )

    except Exception as e:

        return (
            f"❌ Error CNN: {str(e)}",
            None,
            None,
            pd.DataFrame()
        )


# ======================================================================
# HISTORY
# ======================================================================

def history_process(dap_filter):

    try:

        if (
            dap_filter is None
            or str(dap_filter) == ""
            or str(dap_filter) == "Semua Data"
        ):

            data = df_dashboard.copy()

            filter_text = "Semua Data"

        else:

            dap_value = int(
                float(dap_filter)
            )

            data = df_dashboard[
                df_dashboard["DAP"] == dap_value
            ].copy()

            filter_text = f"DAP {dap_value}"

        if len(data) == 0:

            return (
                "❌ Tidak ada data.",
                None,
                pd.DataFrame()
            )

        moisture_mean = data[
            "Soil Moisture (%)"
        ].mean()

        temp_mean = data[
            "Temperature (°C)"
        ].mean()

        maturity_mean = data[
            "Ground Truth Maturity Level (%)"
        ].mean()

        if maturity_mean < 70:
            fase = "Fase Vegetatif"
        elif maturity_mean < 90:
            fase = "Fase Pembentukan Tajuk"
        else:
            fase = "Fase Siap Panen"

        summary = f"""
## 📈 ANALYTICS / HISTORY MONITORING

### Filter: **{filter_text}**

**Jumlah Data:** {len(data)}

### Soil Moisture
- Minimum: **{data["Soil Moisture (%)"].min():.2f} %**
- Maximum: **{data["Soil Moisture (%)"].max():.2f} %**
- Rata-rata: **{moisture_mean:.2f} %**

### Temperature
- Minimum: **{data["Temperature (°C)"].min():.2f} °C**
- Maximum: **{data["Temperature (°C)"].max():.2f} °C**
- Rata-rata: **{temp_mean:.2f} °C**

### Maturity
- Minimum: **{data["Ground Truth Maturity Level (%)"].min():.2f} %**
- Maximum: **{data["Ground Truth Maturity Level (%)"].max():.2f} %**
- Rata-rata: **{maturity_mean:.2f} %**

### Fase
**{fase}**
"""

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data["DAP"],
                y=data[
                    "Ground Truth Maturity Level (%)"
                ],
                mode="lines+markers",
                name="Maturity"
            )
        )

        fig.update_layout(
            title="History Ground Truth Maturity",
            xaxis_title="DAP / HST",
            yaxis_title="Maturity (%)",
            height=450
        )

        return (
            summary,
            fig,
            data
        )

    except Exception as e:

        return (
            f"❌ Error History: {str(e)}",
            None,
            pd.DataFrame()
        )


# ======================================================================
# FULL INTERFACE
# ======================================================================

with gr.Blocks(
    title="Monitoring Pakcoy MILLI"
) as demo:

    gr.Markdown(
        """
# 🌱 SISTEM MONITORING PAKCOY

### Monitoring Sensor • Prediksi LSTM • Klasifikasi CNN
"""
    )


    # ==================================================================
    # DASHBOARD
    # ==================================================================

    with gr.Tab("📊 Dashboard"):

        dashboard_info = gr.Markdown()

        dashboard_graph = gr.Plot(
            label="Monitoring Pakcoy"
        )

        dashboard_index = gr.Slider(
            minimum=0,
            maximum=TOTAL_DATA - 1,
            value=0,
            step=1,
            interactive=False,
            label="Data Monitoring"
        )

        with gr.Row():

            dashboard_start = gr.Button(
                "▶ Start",
                variant="primary"
            )

            dashboard_pause = gr.Button(
                "⏸ Pause"
            )

            dashboard_reset = gr.Button(
                "↩ Reset"
            )

        dashboard_timer = gr.Timer(
            value=2.0,
            active=True
        )

        demo.load(
            dashboard_display,
            inputs=dashboard_index,
            outputs=[
                dashboard_info,
                dashboard_graph
            ]
        )

        dashboard_timer.tick(
            dashboard_next,
            inputs=dashboard_index,
            outputs=dashboard_index
        )

        dashboard_index.change(
            dashboard_display,
            inputs=dashboard_index,
            outputs=[
                dashboard_info,
                dashboard_graph
            ]
        )

        dashboard_start.click(
            lambda: gr.Timer(
                value=2.0,
                active=True
            ),
            outputs=dashboard_timer
        )

        dashboard_pause.click(
            lambda: gr.Timer(
                value=2.0,
                active=False
            ),
            outputs=dashboard_timer
        )

        dashboard_reset.click(
            lambda: 0,
            outputs=dashboard_index
        )


    # ==================================================================
    # LSTM
    # ==================================================================

    with gr.Tab("🔮 LSTM Prediction"):

        gr.Markdown(
            """
## 🔮 Prediksi Maturity Pakcoy 7 Hari
"""
        )

        with gr.Row():

            lstm_dap = gr.Number(
                value=307,
                label="DAP Awal Input",
                precision=0
            )

            lstm_button = gr.Button(
                "🔮 Proses Prediksi",
                variant="primary"
            )

        lstm_summary = gr.Markdown()

        lstm_table = gr.Dataframe(
            label="Hasil Prediksi 7 Hari",
            interactive=False
        )

        lstm_graph = gr.Plot(
            label="Grafik Prediksi LSTM"
        )

        lstm_button.click(
            lstm_prediction,
            inputs=lstm_dap,
            outputs=[
                lstm_summary,
                lstm_table,
                lstm_graph
            ]
        )


    # ==================================================================
    # CNN
    # ==================================================================

    with gr.Tab("🖼️ CNN Classification"):

        gr.Markdown(
            """
## 🖼️ Klasifikasi Fase Pakcoy

Upload gambar tanaman Pakcoy.
"""
        )

        cnn_image = gr.Image(
            type="numpy",
            label="Upload Gambar Pakcoy"
        )

        cnn_button = gr.Button(
            "🔍 Klasifikasi Gambar",
            variant="primary"
        )

        cnn_summary = gr.Markdown()

        cnn_processed = gr.Image(
            label="Gambar Diproses"
        )

        cnn_graph = gr.Plot(
            label="Probabilitas Klasifikasi"
        )

        cnn_table = gr.Dataframe(
            label="Probabilitas Setiap Kelas",
            interactive=False
        )

        cnn_button.click(
            cnn_prediction,
            inputs=cnn_image,
            outputs=[
                cnn_summary,
                cnn_processed,
                cnn_graph,
                cnn_table
            ]
        )


    # ==================================================================
    # HISTORY
    # ==================================================================

    with gr.Tab("📈 Analytics / History"):

        gr.Markdown(
            """
## 📈 Analytics / History Monitoring
"""
        )

        history_filter = gr.Dropdown(
            choices=[
                "Semua Data"
            ] + list(
                range(DAP_MIN, DAP_MAX + 1)
            ),
            value="Semua Data",
            label="Filter DAP"
        )

        history_button = gr.Button(
            "📊 Tampilkan Analytics",
            variant="primary"
        )

        history_summary = gr.Markdown()

        history_graph = gr.Plot(
            label="History Monitoring"
        )

        history_table = gr.Dataframe(
            label="History Data",
            interactive=False
        )

        history_button.click(
            history_process,
            inputs=history_filter,
            outputs=[
                history_summary,
                history_graph,
                history_table
            ]
        )

        demo.load(
            lambda: history_process("Semua Data"),
            inputs=None,
            outputs=[
                history_summary,
                history_graph,
                history_table
            ]
        )


# ======================================================================
# LAUNCH
# ======================================================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get(
                "PORT",
                7860
            )
        )
    )
