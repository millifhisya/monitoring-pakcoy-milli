
# ======================================================================
# 🌱 MONITORING PAKCOY MILLI
# DEPLOYMENT APPLICATION
# ======================================================================
#
# Interface source diambil dari interface aktif Colab.
#
# Training       : TIDAK ADA
# Retraining     : TIDAK ADA
# Model berubah  : TIDAK
#
# Model yang digunakan:
# - Modified LSTM Attention Seq2Seq 7 Days FINAL
# - Modified CNN Spatial Attention FINAL
#
# ======================================================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import tensorflow as tf

from pathlib import Path
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent


# ======================================================================
# FILE PATH
# ======================================================================

DATA_PATH = BASE_DIR / "Dataset_Pakcoy.xlsx"

LSTM_FINAL_PATH = (
    BASE_DIR /
    "Modified_LSTM_Attention_Seq2Seq_7Days_TRUE_FINAL.keras"
)

CNN_FINAL_PATH = (
    BASE_DIR /
    "CNN_Modified_Spatial_Attention_Pakcoy_FINAL.keras"
)

SCALER_X_PATH = (
    BASE_DIR /
    "scaler_X_TRUE_FINAL.pkl"
)

SCALER_Y_PATH = (
    BASE_DIR /
    "scaler_y_TRUE_FINAL.pkl"
)


# ======================================================================
# CUSTOM LSTM LAYER
# ======================================================================

@tf.keras.utils.register_keras_serializable(
    package="Pakcoy"
)
class AttentionContext(tf.keras.layers.Layer):

    def call(self, inputs):

        features = inputs[0]
        attention_weights = inputs[1]

        context = tf.reduce_sum(
            features * attention_weights,
            axis=1
        )

        return context

    def compute_output_shape(
        self,
        input_shape
    ):

        return (
            input_shape[0][0],
            input_shape[0][2]
        )


# ======================================================================
# CUSTOM CNN LAYER
# ======================================================================

@tf.keras.utils.register_keras_serializable(
    package="Pakcoy"
)
class SpatialAttention(tf.keras.layers.Layer):

    def __init__(
        self,
        kernel_size=7,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.kernel_size = kernel_size

        self.conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            padding="same",
            activation="sigmoid",
            use_bias=True
        )

    def call(self, inputs):

        avg_pool = tf.reduce_mean(
            inputs,
            axis=-1,
            keepdims=True
        )

        max_pool = tf.reduce_max(
            inputs,
            axis=-1,
            keepdims=True
        )

        concat = tf.concat(
            [
                avg_pool,
                max_pool
            ],
            axis=-1
        )

        attention = self.conv(concat)

        return inputs * attention

    def get_config(self):

        config = super().get_config()

        config.update({
            "kernel_size": self.kernel_size
        })

        return config


# ======================================================================
# LOAD DATA
# ======================================================================

print("=" * 75)
print("🌱 LOAD MONITORING PAKCOY")
print("=" * 75)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATA_PATH}"
    )

df_preview = pd.read_excel(DATA_PATH)

df_preview = df_preview.copy()

df_preview = df_preview.sort_values(
    ["DAP", "Time"],
    kind="stable"
).reset_index(drop=True)

df_preview["Replay_Index"] = np.arange(
    len(df_preview)
)

print(
    f"✅ Dataset : {len(df_preview)} data"
)

print(
    f"✅ DAP     : "
    f"{int(df_preview['DAP'].min())} - "
    f"{int(df_preview['DAP'].max())}"
)


# ======================================================================
# LOAD SCALER
# ======================================================================

with open(
    SCALER_X_PATH,
    "rb"
) as f:

    scaler_X_preview = pickle.load(f)


with open(
    SCALER_Y_PATH,
    "rb"
) as f:

    scaler_y_preview = pickle.load(f)


print("✅ Scaler X loaded")
print("✅ Scaler Y loaded")


# ======================================================================
# LOAD LSTM FINAL
# ======================================================================

lstm_preview = tf.keras.models.load_model(
    LSTM_FINAL_PATH,
    custom_objects={
        "AttentionContext": AttentionContext
    },
    compile=False,
    safe_mode=True
)

print(
    f"✅ LSTM FINAL loaded "
    f"({lstm_preview.count_params():,} parameters)"
)


# ======================================================================
# LOAD CNN FINAL
# ======================================================================

cnn_preview = tf.keras.models.load_model(
    CNN_FINAL_PATH,
    custom_objects={
        "SpatialAttention": SpatialAttention
    },
    compile=False,
    safe_mode=True
)

print(
    f"✅ CNN FINAL loaded "
    f"({cnn_preview.count_params():,} parameters)"
)


# ======================================================================
# ALIAS UNTUK INTERFACE AKTIF
# ======================================================================

df_ui = df_preview.copy()

df_ui = df_ui.sort_values(
    ["DAP", "Time"],
    kind="stable"
).reset_index(drop=True)

df_ui["Replay_Index"] = np.arange(
    len(df_ui)
)

TOTAL_DATA = len(df_ui)

DAP_MIN = int(
    df_ui["DAP"].min()
)

DAP_MAX = int(
    df_ui["DAP"].max()
)

print(
    f"✅ Interface dataset : {TOTAL_DATA}"
)

print(
    f"✅ DAP range         : "
    f"{DAP_MIN} - {DAP_MAX}"
)

print()
print("🚫 Training    : TIDAK ADA")
print("🚫 Retraining  : TIDAK ADA")
print("🚫 Model ubah  : TIDAK")
print("=" * 75)



# ============================================================
# ACTIVE INTERFACE FUNCTIONS
# ============================================================

def maturity_phase(
    maturity
):

    maturity = float(
        maturity
    )

    if maturity < 70:

        return "Fase Vegetatif"

    elif maturity < 90:

        return "Fase Pembentukan Tajuk"

    else:

        return "Fase Siap Panen"



# ============================================================
# FUNCTION: dashboard_render
# ============================================================
def dashboard_render(
    index
):

    index = int(index)

    index = max(
        0,
        min(
            index,
            TOTAL_DATA - 1
        )
    )

    row = df_ui.iloc[
        index
    ]

    maturity = float(
        row[
            "Ground Truth Maturity Level (%)"
        ]
    )

    phase = maturity_phase(
        maturity
    )

    start = max(
        0,
        index - 29
    )

    hist = df_ui.iloc[
        start:index + 1
    ].copy()

    fig, ax1 = plt.subplots(
        figsize=(11, 5)
    )

    ax1.plot(
        hist["Replay_Index"],
        hist["Soil Moisture (%)"],
        label="Soil Moisture (%)"
    )

    ax1.plot(
        hist["Replay_Index"],
        hist["Temperature (°C)"],
        label="Temperature (°C)"
    )

    ax1.set_xlabel(
        "Replay Data"
    )

    ax1.set_ylabel(
        "Sensor Value"
    )

    ax1.grid(
        True,
        alpha=0.3
    )

    ax2 = ax1.twinx()

    ax2.plot(
        hist["Replay_Index"],
        hist[
            "Ground Truth Maturity Level (%)"
        ],
        label="Maturity (%)"
    )

    ax2.set_ylabel(
        "Maturity (%)"
    )

    lines1, labels1 = (
        ax1.get_legend_handles_labels()
    )

    lines2, labels2 = (
        ax2.get_legend_handles_labels()
    )

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left"
    )

    ax1.set_title(
        "Monitoring Pakcoy — Replay Data"
    )

    plt.tight_layout()

    day_text = (
        str(row["Day"])
        if "Day" in row.index
        else "-"
    )

    info = f"""
# 🌱 Monitoring Pakcoy

### Data {index + 1} / {TOTAL_DATA}

| Parameter | Nilai |
|---|---:|
| **Day** | {day_text} |
| **DAP / HST** | {int(row["DAP"])} |
| **Time** | {row["Time"]} |
| **Soil Moisture** | {float(row["Soil Moisture (%)"]):.2f}% |
| **Temperature** | {float(row["Temperature (°C)"]):.2f} °C |
| **Maturity** | {maturity:.2f}% |
| **Fase** | **{phase}** |

**Status:** 🟢 Monitoring aktif
"""

    return (
        info,
        fig
    )



# ============================================================
# FUNCTION: dashboard_next
# ============================================================
def dashboard_next(
    index
):

    index = int(index)

    next_index = index + 1

    if next_index >= TOTAL_DATA:

        next_index = 0

    return next_index



# ============================================================
# FUNCTION: lstm_predict_interface
# ============================================================
def lstm_predict_interface(
    dap_value
):

    try:

        dap_value = int(
            float(dap_value)
        )

    except Exception:

        return (
            "❌ DAP tidak valid.",
            pd.DataFrame(),
            None
        )

    available = df_ui[
        df_ui["DAP"] <= dap_value
    ].copy()

    if len(available) < TIME_STEPS:

        return (
            f"""
### ⚠️ Data belum cukup

LSTM membutuhkan **21 observasi terakhir**.

DAP yang dipilih: **{dap_value}**

Observasi tersedia:
**{len(available)}**
""",
            pd.DataFrame(),
            None
        )

    sequence_df = (
        available[
            LSTM_FEATURES
        ]
        .tail(TIME_STEPS)
        .copy()
    )

    sequence = sequence_df.values.astype(
        np.float32
    )

    # SCALING INPUT
    X = scaler_X_preview.transform(
        sequence
    )

    X = X.reshape(
        1,
        TIME_STEPS,
        len(LSTM_FEATURES)
    )

    # PREDIKSI
    prediction_scaled = (
        lstm_preview.predict(
            X,
            verbose=0
        )
    )

    prediction_scaled = np.asarray(
        prediction_scaled
    ).reshape(
        -1,
        1
    )

    # INVERSE SCALING
    prediction = (
        scaler_y_preview
        .inverse_transform(
            prediction_scaled
        )
        .reshape(-1)
    )

    prediction = np.clip(
        prediction,
        0,
        100
    )

    # ==============================================================
    # WAKTU
    # ==============================================================

    times = (
        df_ui["Time"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(times) >= 3:

        times = times[:3]

    else:

        times = [
            "08:00",
            "12:00",
            "16:00"
        ]

    future_rows = []

    for i in range(
        FORECAST_STEPS
    ):

        future_day = (
            dap_value
            + (i // 3)
            + 1
        )

        future_time = times[
            i % 3
        ]

        future_rows.append({

            "Hari ke":
                (i // 3) + 1,

            "DAP/HST":
                future_day,

            "Waktu":
                future_time,

            "Prediksi Maturity (%)":
                round(
                    float(
                        prediction[i]
                    ),
                    2
                ),

            "Fase":
                maturity_phase(
                    prediction[i]
                )
        })

    forecast_df = pd.DataFrame(
        future_rows
    )

    # ==============================================================
    # GRAPH
    # ==============================================================

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        range(
            1,
            FORECAST_STEPS + 1
        ),
        prediction,
        marker="o",
        label="LSTM Prediction"
    )

    ax.set_xlabel(
        "Observasi Forecast"
    )

    ax.set_ylabel(
        "Predicted Maturity (%)"
    )

    ax.set_title(
        f"LSTM Forecast 7 Hari — "
        f"mulai DAP {dap_value}"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    first_pred = float(
        prediction[0]
    )

    last_pred = float(
        prediction[-1]
    )

    info = f"""
# 🔮 LSTM Prediction

**DAP saat ini:** {dap_value}

**Forecast:** 7 hari / 21 observasi

**Prediksi awal:** {first_pred:.2f}%

**Prediksi akhir:** {last_pred:.2f}%

**Perubahan:** {last_pred - first_pred:+.2f}%

> Model: **Modified LSTM + Temporal Attention + Seq2Seq**

### Input yang digunakan

- DAP
- Soil Moisture (%)
- Temperature (°C)

**Input sequence:** 21 observasi
"""

    return (
        info,
        forecast_df,
        fig
    )



# ============================================================
# FUNCTION: cnn_predict_interface
# ============================================================
def cnn_predict_interface(
    image
):

    if image is None:

        return (
            "⚠️ Silakan upload foto tanaman Pakcoy.",
            pd.DataFrame()
        )

    try:

        if not isinstance(
            image,
            Image.Image
        ):

            image = Image.fromarray(
                np.asarray(image)
            )

        image = image.convert(
            "RGB"
        )

        image = image.resize(
            (224, 224)
        )

        arr = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        arr = np.expand_dims(
            arr,
            axis=0
        )

        probabilities = (
            cnn_preview.predict(
                arr,
                verbose=0
            )[0]
        )

        predicted_class = int(
            np.argmax(
                probabilities
            )
        )

        confidence = (
            float(
                probabilities[
                    predicted_class
                ]
            ) * 100
        )

        rows = []

        for class_id, prob in enumerate(
            probabilities
        ):

            rows.append({

                "Kelas":
                    CNN_CLASSES[
                        class_id
                    ],

                "Status":
                    CNN_DISPLAY[
                        class_id
                    ],

                "Probabilitas (%)":
                    round(
                        float(prob) * 100,
                        2
                    )
            })

        result_df = pd.DataFrame(
            rows
        )

        if confidence >= 80:

            level = "TINGGI"

        elif confidence >= 60:

            level = "SEDANG"

        else:

            level = "RENDAH"

        info = f"""
# 🖼️ CNN Classification

### Hasil

**Kelas:**
{CNN_CLASSES[predicted_class]}

**Status:**
**{CNN_DISPLAY[predicted_class]}**

**Confidence:**
**{confidence:.2f}%**

**Tingkat keyakinan:**
**{level}**

Model:
**Modified CNN + Spatial Attention**
"""

        return (
            info,
            result_df
        )

    except Exception as e:

        return (
            f"❌ Gagal memproses gambar: `{str(e)}`",
            pd.DataFrame()
        )



# ============================================================
# FUNCTION: analytics_history_interface
# ============================================================
def analytics_history_interface(
    dap_filter="Semua Data"
):

    if (
        dap_filter is None
        or str(dap_filter) == "Semua Data"
    ):

        data = df_ui.copy()

        filter_text = "Semua Data"

    else:

        dap = int(
            float(dap_filter)
        )

        data = df_ui[
            df_ui["DAP"] == dap
        ].copy()

        filter_text = (
            f"DAP {dap}"
        )

    if len(data) == 0:

        return (
            "❌ Tidak ada data.",
            pd.DataFrame(),
            None
        )

    mean_moisture = (
        data["Soil Moisture (%)"]
        .mean()
    )

    mean_temp = (
        data["Temperature (°C)"]
        .mean()
    )

    mean_maturity = (
        data[
            "Ground Truth Maturity Level (%)"
        ].mean()
    )

    max_maturity = (
        data[
            "Ground Truth Maturity Level (%)"
        ].max()
    )

    info = f"""
# 📈 Analytics / History

**Filter:** {filter_text}

| Statistik | Nilai |
|---|---:|
| Jumlah data | {len(data)} |
| DAP minimum | {int(data["DAP"].min())} |
| DAP maksimum | {int(data["DAP"].max())} |
| Rata-rata Soil Moisture | {mean_moisture:.2f}% |
| Rata-rata Temperature | {mean_temp:.2f} °C |
| Rata-rata Maturity | {mean_maturity:.2f}% |
| Maturity maksimum | {max_maturity:.2f}% |
"""

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        data["DAP"],
        data[
            "Ground Truth Maturity Level (%)"
        ],
        label="Maturity (%)"
    )

    ax.set_xlabel(
        "DAP / HST"
    )

    ax.set_ylabel(
        "Maturity (%)"
    )

    ax.set_title(
        f"History Maturity — {filter_text}"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    display_columns = [
        "Day",
        "DAP",
        "Time",
        "Soil Moisture (%)",
        "Temperature (°C)",
        "Ground Truth Maturity Level (%)"
    ]

    display_columns = [
        c
        for c in display_columns
        if c in data.columns
    ]

    table = data[
        display_columns
    ].copy()

    return (
        info,
        table,
        fig
    )

# ======================================================================

# ======================================================================
# ACTIVE GRADIO INTERFACE
# ======================================================================

with gr.Blocks(
    title="Sistem Monitoring Pakcoy"
) as demo:

    gr.Markdown(
        """
# 🌱 SISTEM MONITORING PAKCOY

### Monitoring Sensor • Prediksi LSTM • Klasifikasi CNN • Analytics
"""
    )


    # ==============================================================
    # DASHBOARD
    # ==============================================================

    with gr.Tab(
        "📊 Dashboard"
    ):

        dashboard_info = (
            gr.Markdown()
        )

        dashboard_graph = gr.Plot(
            label="Monitoring Pakcoy"
        )

        dashboard_index = gr.Slider(
            minimum=0,
            maximum=TOTAL_DATA - 1,
            value=0,
            step=1,
            label="Replay Data",
            interactive=True
        )

        gr.Markdown(
            "▶️ Dashboard bergerak otomatis setiap 1 detik."
        )

        dashboard_timer = gr.Timer(
            value=1.0,
            active=True
        )

        dashboard_timer.tick(
            dashboard_next,
            inputs=dashboard_index,
            outputs=dashboard_index
        )

        dashboard_index.change(
            dashboard_render,
            inputs=dashboard_index,
            outputs=[
                dashboard_info,
                dashboard_graph
            ]
        )

        demo.load(
            dashboard_render,
            inputs=dashboard_index,
            outputs=[
                dashboard_info,
                dashboard_graph
            ]
        )


    # ==============================================================
    # LSTM
    # ==============================================================

    with gr.Tab(
        "🔮 LSTM Prediction"
    ):

        gr.Markdown(
            """
## Prediksi Maturity Pakcoy 7 Hari

Pilih **DAP/HST saat ini**, kemudian sistem menggunakan
**21 observasi terakhir** untuk menghasilkan
**21 observasi forecast (3 observasi/hari = 7 hari).**

Input LSTM:
**DAP + Soil Moisture + Temperature**
"""
        )

        lstm_dap = gr.Dropdown(
            choices=DAP_CHOICES,
            value=DEFAULT_DAP,
            label="DAP / HST Saat Ini"
        )

        lstm_button = gr.Button(
            "🔮 Prediksi 7 Hari",
            variant="primary"
        )

        lstm_info = gr.Markdown()

        lstm_table = gr.Dataframe(
            label="Hasil Forecast",
            interactive=False
        )

        lstm_graph = gr.Plot(
            label="LSTM Forecast"
        )

        lstm_button.click(
            lstm_predict_interface,
            inputs=lstm_dap,
            outputs=[
                lstm_info,
                lstm_table,
                lstm_graph
            ]
        )


    # ==============================================================
    # CNN
    # ==============================================================

    with gr.Tab(
        "🖼️ CNN Classification"
    ):

        gr.Markdown(
            """
## Penilaian Kondisi Tanaman Pakcoy

Upload foto tanaman untuk memperoleh klasifikasi:

- **Masa Pertumbuhan**
- **Mendekati Panen**
- **Siap Panen**
"""
        )

        with gr.Row():

            cnn_image = gr.Image(
                type="pil",
                label="Upload Foto Pakcoy"
            )

            cnn_result = gr.Markdown()

        cnn_button = gr.Button(
            "🔍 Analisis Foto",
            variant="primary"
        )

        cnn_probability = gr.Dataframe(
            label="Probabilitas Kelas",
            interactive=False
        )

        cnn_button.click(
            cnn_predict_interface,
            inputs=cnn_image,
            outputs=[
                cnn_result,
                cnn_probability
            ]
        )


    # ==============================================================
    # ANALYTICS
    # ==============================================================

    with gr.Tab(
        "📈 Analytics / History"
    ):

        history_filter = gr.Dropdown(
            choices=[
                "Semua Data"
            ] + list(
                range(
                    DAP_MIN,
                    DAP_MAX + 1
                )
            ),
            value="Semua Data",
            label="Filter DAP / HST"
        )

        history_button = gr.Button(
            "📊 Tampilkan History",
            variant="primary"
        )

        history_info = gr.Markdown()

        history_table = gr.Dataframe(
            label="Data History",
            interactive=False
        )

        history_graph = gr.Plot(
            label="Grafik History"
        )

        history_button.click(
            analytics_history_interface,
            inputs=history_filter,
            outputs=[
                history_info,
                history_table,
                history_graph
            ]
        )


# ======================================================================
# 9. STATUS
# ======================================================================

print("=" * 75)
print("✅ INTERFACE BERHASIL DIBANGUN")
print("=" * 75)

print(
    f"Dataset   : {TOTAL_DATA} data"
)

print(
    f"DAP       : {DAP_MIN} - {DAP_MAX}"
)

print(
    f"LSTM      : "
    f"{lstm_preview.count_params():,} parameters"
)

print(
    f"CNN       : "
    f"{cnn_preview.count_params():,} parameters"
)

print("Dashboard : READY")
print("LSTM      : READY")
print("CNN       : READY")
print("Analytics : READY")

print()
print("🚫 Training   : TIDAK ADA")
print("🚫 Retraining : TIDAK ADA")
print("🚫 Model      : TIDAK DIUBAH")

print("=" * 75)


# ======================================================================
# 10. LAUNCH PREVIEW
# ======================================================================

# ======================================================================
# RENDER DEPLOYMENT LAUNCH
# ======================================================================

if __name__ == "__main__":

    print("=" * 75)
    print("🌱 MONITORING PAKCOY MILLI")
    print("=" * 75)
    print("Dashboard       : READY")
    print("LSTM Prediction : READY")
    print("CNN             : READY")
    print("Analytics       : READY")
    print("Training        : TIDAK ADA")
    print("Retraining      : TIDAK ADA")
    print("Model change    : TIDAK ADA")
    print("=" * 75)

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get("PORT", "7860")
        ),
        share=False,
        debug=False,
        show_error=True
    )
