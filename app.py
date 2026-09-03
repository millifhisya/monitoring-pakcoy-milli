import os

# ============================================================================
# STREAMLIT CLOUD - MEMORY OPTIMIZATION
# ============================================================================

# Paksa TensorFlow menggunakan CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Batasi thread CPU agar penggunaan resource lebih ringan
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import time
import pickle
import json
import gc

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================================
# AUTOREFRESH
# ============================================================================

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except Exception:
    AUTOREFRESH_AVAILABLE = False


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Monitoring Pakcoy MILLI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CATATAN TENSORFLOW
# ============================================================================
#
# TensorFlow SENGAJA TIDAK di-import di sini.
#
# TensorFlow akan dimuat hanya ketika fungsi LSTM/CNN dipanggil.
# Tujuannya agar Dashboard tidak langsung memakan RAM besar.
#
# Custom CNN Layer juga akan dibuat saat CNN benar-benar digunakan.
#
# ============================================================================


# ============================================================================
# PATH
# ============================================================================

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


# ============================================================================
# LOAD DATASET
# ============================================================================

@st.cache_data
def load_dataset():

    df = pd.read_excel(DATASET_PATH)

    # Time ordering
    time_order = {
        "08:00": 1,
        "12:00": 2,
        "16:00": 3
    }

    df["_Time_Order"] = (
        df["Time"]
        .astype(str)
        .str[:5]
        .map(time_order)
    )

    df = df.sort_values(
        ["Day", "DAP", "_Time_Order"]
    ).reset_index(drop=True)

    df["Replay_Index"] = np.arange(len(df))

    # Timestamp
    if "Timestamp" not in df.columns:

        base_date = pd.Timestamp("2026-09-01")

        df["Timestamp"] = [
            base_date
            + pd.Timedelta(days=int(day) - 1)
            + pd.Timedelta(
                hours=int(str(t)[:2])
            )
            for day, t in zip(
                df["Day"],
                df["Time"]
            )
        ]

    return df


df = load_dataset()


# ============================================================================
# LOAD LSTM - LAZY LOADING
# ============================================================================

@st.cache_resource
def load_lstm():

    import tensorflow as tf
    from tensorflow.keras.models import load_model

    model = load_model(
        LSTM_MODEL_PATH,
        compile=False
    )

    with open(
        LSTM_SCALER_PATH,
        "rb"
    ) as f:
        scaler = pickle.load(f)

    return model, scaler


# ============================================================================
# LOAD CNN - LAZY LOADING
# ============================================================================

@st.cache_resource
def load_cnn():

    import tensorflow as tf
    from tensorflow.keras.models import load_model

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

    model = load_model(
        CNN_MODEL_PATH,
        compile=False,
        custom_objects={
            "ChannelAveragePool": ChannelAveragePool,
            "ChannelMaxPool": ChannelMaxPool
        }
    )

    return model


# ============================================================================
# PHASE
# ============================================================================
def get_phase(maturity):

    maturity = float(maturity)

    if maturity < 70:
        return "Fase Vegetatif"

    elif maturity < 90:
        return "Fase Pembentukan Tajuk"

    return "Fase Siap Panen"


# ============================================================================
# DASHBOARD DATA
# ============================================================================

def get_dashboard_row(index):

    index = int(index)

    row = df.iloc[index]

    return row


# ============================================================================
# DASHBOARD PLOT
# ============================================================================

def dashboard_plot(index):

    index = int(index)

    history = df.iloc[:index + 1]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 9)
    )

    # Soil moisture
    axes[0].plot(
        history["Replay_Index"],
        history["Soil Moisture (%)"]
    )

    axes[0].set_title(
        "Soil Moisture (%)"
    )

    axes[0].set_ylabel("%")
    axes[0].grid(True, alpha=0.3)

    # Temperature
    axes[1].plot(
        history["Replay_Index"],
        history["Temperature (°C)"]
    )

    axes[1].set_title(
        "Temperature (°C)"
    )

    axes[1].set_ylabel("°C")
    axes[1].grid(True, alpha=0.3)

    # Maturity
    axes[2].plot(
        history["Replay_Index"],
        history[
            "Ground Truth Maturity Level (%)"
        ]
    )

    axes[2].set_title(
        "Ground Truth Maturity Level (%)"
    )

    axes[2].set_xlabel(
        "Replay Index"
    )

    axes[2].set_ylabel("%")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


# ============================================================================
# LSTM PREDICTION
# ============================================================================

def predict_lstm(dap_start):

    dap_start = int(dap_start)

    # Ambil 21 observasi = 7 hari × 3 waktu
    input_df = df[
        df["DAP"].between(
            dap_start,
            dap_start + 6
        )
    ].copy()

    input_df = input_df.sort_values(
        ["DAP", "_Time_Order"]
    )

    if len(input_df) != 21:

        raise ValueError(
            "Data input LSTM harus memiliki "
            "21 observasi."
        )

    features = [
        "DAP",
        "Soil Moisture (%)",
        "Temperature (°C)"
    ]

    X = (
        input_df[features]
        .astype(np.float32)
        .values
    )

    X = np.expand_dims(
        X,
        axis=0
    )

    model, scaler = load_lstm()

    pred_scaled = model.predict(
        X,
        verbose=0
    )

    pred_scaled = np.asarray(
        pred_scaled
    ).reshape(-1, 1)

    predictions = scaler.inverse_transform(
        pred_scaled
    ).flatten()

    # Future 7 hari
    future_daps = []

    for day_offset in range(7):

        future_dap = dap_start + 7 + day_offset

        for waktu in [
            "08:00",
            "12:00",
            "16:00"
        ]:

            future_daps.append(
                (
                    future_dap,
                    waktu
                )
            )

    # Model menghasilkan 21 output
    result = pd.DataFrame({
        "Horizon": [
            f"H{i // 3 + 1}"
            for i in range(21)
        ],
        "DAP": [
            x[0]
            for x in future_daps
        ],
        "Time": [
            x[1]
            for x in future_daps
        ],
        "Prediksi Maturity (%)": np.round(
            predictions,
            2
        )
    })

    result["Fase"] = result[
        "Prediksi Maturity (%)"
    ].apply(get_phase)

    return result


# ============================================================================
# CNN
# ============================================================================

CLASS_NAMES = [
    "Fase_Vegetatif",
    "Fase_Pembentukan_Tajuk",
    "Fase_Siap_Panen"
]


def predict_cnn(uploaded_file):

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_resized = image.resize(
        (224, 224)
    )

    array = np.asarray(
        image_resized,
        dtype=np.float32
    ) / 255.0

    batch = np.expand_dims(
        array,
        axis=0
    )

    model = load_cnn()

    probabilities = model.predict(
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
        probabilities[predicted_index]
    )

    result = pd.DataFrame({
        "Kelas": CLASS_NAMES,
        "Confidence (%)": np.round(
            probabilities * 100,
            2
        )
    })

    return (
        image,
        predicted_class,
        confidence,
        result
    )


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title(
    "🌱 Monitoring Pakcoy"
)

st.sidebar.markdown(
    """
    **Sistem Monitoring Pakcoy**

    Dashboard berbasis dataset,
    prediksi LSTM, klasifikasi CNN,
    dan riwayat data.
    """
)

st.sidebar.divider()

st.sidebar.info(
    f"Dataset: {len(df):,} data"
)


# ============================================================================
# TITLE
# ============================================================================

st.title(
    "🌱 SISTEM MONITORING PAKCOY"
)

st.caption(
    "Monitoring Dashboard • LSTM Prediction • CNN Classification • Analytics / History"
)


# ============================================================================
# TABS
# ============================================================================

tab_dashboard, tab_lstm, tab_cnn, tab_history = st.tabs(
    [
        "🌱 Dashboard",
        "🧠 LSTM Prediction",
        "📷 CNN Classification",
        "📊 Analytics / History"
    ]
)


# ============================================================================
# 1. DASHBOARD
# ============================================================================

with tab_dashboard:

    st.header(
        "🌱 Dashboard Monitoring"
    )

    st.write(
        "Replay dataset dari data pertama "
        "sampai data terakhir."
    )

    # Session state
    if "dashboard_index" not in st.session_state:

        st.session_state.dashboard_index = 0

    autoplay = st.checkbox(
        "▶️ Autoplay",
        value=True
    )

    interval_ms = st.slider(
        "Kecepatan autoplay (ms)",
        min_value=300,
        max_value=3000,
        value=1000,
        step=100
    )

    if (
        autoplay
        and AUTOREFRESH_AVAILABLE
    ):

        st_autorefresh(
            interval=interval_ms,
            key="dashboard_refresh"
        )

        st.session_state.dashboard_index = (
            st.session_state.dashboard_index + 1
        ) % len(df)

    elif autoplay and not AUTOREFRESH_AVAILABLE:

        st.warning(
            "Autoplay membutuhkan "
            "streamlit-autorefresh."
        )

    index = st.slider(
        "Replay Dataset",
        min_value=0,
        max_value=len(df) - 1,
        value=int(
            st.session_state.dashboard_index
        ),
        key="dashboard_slider"
    )

    if index != st.session_state.dashboard_index:

        st.session_state.dashboard_index = index

    row = get_dashboard_row(index)

    phase = get_phase(
        row[
            "Ground Truth Maturity Level (%)"
        ]
    )

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Data",
        f"{index + 1:,} / {len(df):,}"
    )

    c2.metric(
        "Day",
        int(row["Day"])
    )

    c3.metric(
        "DAP / HST",
        int(row["DAP"])
    )

    c4.metric(
        "Soil Moisture",
        f'{row["Soil Moisture (%)"]:.0f}%'
    )

    c5.metric(
        "Temperature",
        f'{row["Temperature (°C)"]:.1f}°C'
    )

    st.subheader(
        "Informasi Data Saat Ini"
    )

    info_df = pd.DataFrame({
        "Parameter": [
            "Tanggal",
            "Waktu",
            "Soil Condition",
            "Ground Truth Maturity",
            "Fase",
            "Ground Truth Criteria"
        ],
        "Nilai": [
            str(
                row["Timestamp"]
            )[:10],
            str(
                row["Time"]
            ),
            row["Soil Condition"],
            f'{row["Ground Truth Maturity Level (%)"]:.2f}%',
            phase,
            row["Ground Truth Criteria"]
        ]
    })

    st.table(info_df)

    st.pyplot(
        dashboard_plot(index),
        use_container_width=True
    )


# ============================================================================
# 2. LSTM
# ============================================================================

with tab_lstm:

    st.header(
        "🧠 Prediksi LSTM 7 Hari"
    )

    st.write(
        "Gunakan 21 observasi terakhir "
        "(7 hari × 3 waktu) untuk memprediksi "
        "21 observasi berikutnya."
    )

    min_dap = int(
        df["DAP"].min()
    )

    max_dap = int(
        df["DAP"].max() - 6
    )

    dap_start = st.number_input(
        "DAP Awal Input",
        min_value=min_dap,
        max_value=max_dap,
        value=min_dap,
        step=1
    )

    if st.button(
        "🚀 Proses Prediksi LSTM",
        type="primary"
    ):

        try:

            with st.spinner(
                "Memproses LSTM..."
            ):

                result = predict_lstm(
                    dap_start
                )

            st.success(
                "Prediksi LSTM berhasil."
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Jumlah Prediksi",
                len(result)
            )

            c2.metric(
                "Prediksi Minimum",
                f'{result["Prediksi Maturity (%)"].min():.2f}%'
            )

            c3.metric(
                "Prediksi Maksimum",
                f'{result["Prediksi Maturity (%)"].max():.2f}%'
            )

            st.subheader(
                "Tabel Prediksi 7 Hari"
            )

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )

            fig, ax = plt.subplots(
                figsize=(12, 5)
            )

            ax.plot(
                range(1, len(result) + 1),
                result[
                    "Prediksi Maturity (%)"
                ],
                marker="o"
            )

            ax.set_xlabel(
                "Observasi Prediksi"
            )

            ax.set_ylabel(
                "Maturity (%)"
            )

            ax.set_title(
                "Prediksi Maturity LSTM - 7 Hari"
            )

            ax.grid(
                True,
                alpha=0.3
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Terjadi error LSTM: {e}"
            )


# ============================================================================
# 3. CNN
# ============================================================================

with tab_cnn:

    st.header(
        "📷 CNN Classification"
    )

    st.write(
        "Upload foto tanaman Pakcoy "
        "untuk melakukan klasifikasi kondisi "
        "pertumbuhan."
    )

    uploaded_file = st.file_uploader(
        "📤 Upload foto Pakcoy",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        try:

            image, predicted_class, confidence, result = (
                predict_cnn(
                    uploaded_file
                )
            )

            c1, c2 = st.columns(
                [1, 1]
            )

            with c1:

                st.image(
                    image,
                    caption="Foto Pakcoy",
                    use_container_width=True
                )

            with c2:

                st.subheader(
                    "Hasil Klasifikasi"
                )

                st.success(
                    predicted_class
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True
                )

            st.subheader(
                "Grafik Confidence"
            )

            fig, ax = plt.subplots(
                figsize=(10, 4)
            )

            ax.bar(
                result["Kelas"],
                result["Confidence (%)"]
            )

            ax.set_ylabel(
                "Confidence (%)"
            )

            ax.set_xlabel(
                "Kelas"
            )

            ax.set_ylim(
                0,
                100
            )

            ax.tick_params(
                axis="x",
                rotation=20
            )

            ax.grid(
                axis="y",
                alpha=0.3
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Terjadi error CNN: {e}"
            )


# ============================================================================
# 4. ANALYTICS / HISTORY
# ============================================================================

with tab_history:

    st.header(
        "📊 Analytics / History"
    )

    st.write(
        "Riwayat seluruh 1.140 data monitoring."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Data",
        f"{len(df):,}"
    )

    c2.metric(
        "DAP Minimum",
        int(df["DAP"].min())
    )

    c3.metric(
        "DAP Maksimum",
        int(df["DAP"].max())
    )

    c4.metric(
        "Jumlah Hari",
        int(df["Day"].nunique())
    )

    st.subheader(
        "Filter DAP"
    )

    dap_filter = st.slider(
        "Pilih DAP",
        min_value=int(
            df["DAP"].min()
        ),
        max_value=int(
            df["DAP"].max()
        ),
        value=int(
            df["DAP"].min()
        )
    )

    filtered = df[
        df["DAP"] == dap_filter
    ].copy()

    st.subheader(
        f"History DAP {dap_filter}"
    )

    st.dataframe(
        filtered.drop(
            columns=["_Time_Order"],
            errors="ignore"
        ),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Statistik Dataset"
    )

    stat_df = pd.DataFrame({
        "Parameter": [
            "Soil Moisture (%)",
            "Temperature (°C)",
            "Ground Truth Maturity (%)"
        ],
        "Minimum": [
            df[
                "Soil Moisture (%)"
            ].min(),
            df[
                "Temperature (°C)"
            ].min(),
            df[
                "Ground Truth Maturity Level (%)"
            ].min()
        ],
        "Rata-rata": [
            df[
                "Soil Moisture (%)"
            ].mean(),
            df[
                "Temperature (°C)"
            ].mean(),
            df[
                "Ground Truth Maturity Level (%)"
            ].mean()
        ],
        "Maksimum": [
            df[
                "Soil Moisture (%)"
            ].max(),
            df[
                "Temperature (°C)"
            ].max(),
            df[
                "Ground Truth Maturity Level (%)"
            ].max()
        ]
    })

    st.dataframe(
        stat_df.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Grafik History Maturity"
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        df["DAP"],
        df[
            "Ground Truth Maturity Level (%)"
        ]
    )

    ax.set_xlabel(
        "DAP / HST"
    )

    ax.set_ylabel(
        "Maturity (%)"
    )

    ax.set_title(
        "History Ground Truth Maturity"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    st.subheader(
        "History Soil Moisture"
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        df["DAP"],
        df["Soil Moisture (%)"]
    )

    ax.set_xlabel(
        "DAP / HST"
    )

    ax.set_ylabel(
        "Soil Moisture (%)"
    )

    ax.set_title(
        "History Soil Moisture"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    st.subheader(
        "History Temperature"
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        df["DAP"],
        df["Temperature (°C)"]
    )

    ax.set_xlabel(
        "DAP / HST"
    )

    ax.set_ylabel(
        "Temperature (°C)"
    )

    ax.set_title(
        "History Temperature"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    st.pyplot(
        fig,
        use_container_width=True
    )


# ============================================================================
# FOOTER
# ============================================================================

st.divider()

st.caption(
    "Monitoring Pakcoy MILLI | "
    "Dashboard + LSTM + CNN + Analytics / History"
)
