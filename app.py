import os

# ============================================================================
# STREAMLIT CLOUD - MEMORY OPTIMIZED FINAL
# ============================================================================

# TensorFlow dipaksa CPU dan thread dibatasi agar penggunaan RAM/CPU lebih ringan.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import gc
import json
import pickle
import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Monitoring Pakcoy MILLI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# PATH
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "Dataset_Pakcoy.xlsx")

LSTM_MODEL_PATH = os.path.join(
    BASE_DIR,
    "Modified_LSTM_Seq2Seq_7Days_FINAL.keras",
)

LSTM_SCALER_PATH = os.path.join(
    BASE_DIR,
    "scaler_y_pakcoy_FINAL.pkl",
)

CNN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "CNN_Modified_Spatial_Attention_Pakcoy_FINAL.keras",
)


# ============================================================================
# CHECK FILE
# ============================================================================

def check_required_files():
    required = {
        "Dataset": DATASET_PATH,
        "Model LSTM": LSTM_MODEL_PATH,
        "Scaler LSTM": LSTM_SCALER_PATH,
        "Model CNN": CNN_MODEL_PATH,
    }

    missing = [
        f"{name}: {path}"
        for name, path in required.items()
        if not os.path.exists(path)
    ]

    if missing:
        st.error("File yang diperlukan tidak ditemukan:")
        for item in missing:
            st.write(f"- {item}")
        st.stop()


check_required_files()


# ============================================================================
# DATASET
# ============================================================================

@st.cache_data(show_spinner=False)
def load_dataset():
    data = pd.read_excel(DATASET_PATH)

    required_columns = [
        "Day",
        "DAP",
        "Time",
        "Soil Moisture (%)",
        "Temperature (°C)",
        "Soil Condition",
        "Ground Truth Maturity Level (%)",
        "Ground Truth Criteria",
    ]

    missing = [c for c in required_columns if c not in data.columns]
    if missing:
        raise ValueError(
            "Kolom dataset tidak lengkap: " + ", ".join(missing)
        )

    time_order = {
        "08:00": 1,
        "12:00": 2,
        "16:00": 3,
    }

    data["_Time_Order"] = (
        data["Time"]
        .astype(str)
        .str[:5]
        .map(time_order)
    )

    if data["_Time_Order"].isna().any():
        raise ValueError(
            "Terdapat format Time selain 08:00, 12:00, atau 16:00."
        )

    data = (
        data
        .sort_values(["Day", "DAP", "_Time_Order"])
        .reset_index(drop=True)
    )

    data["Replay_Index"] = np.arange(len(data))

    # Dataset asli tidak memiliki Timestamp.
    # Timestamp dibuat berdasarkan Day + Time.
    if "Timestamp" not in data.columns:
        base_date = pd.Timestamp("2026-09-01")

        data["Timestamp"] = [
            base_date
            + pd.Timedelta(days=int(day) - 1)
            + pd.Timedelta(hours=int(str(t)[:2]))
            for day, t in zip(data["Day"], data["Time"])
        ]

    return data


try:
    df = load_dataset()
except Exception as e:
    st.error(f"Gagal membaca dataset: {e}")
    st.stop()


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
# TENSORFLOW HELPER
# ============================================================================

def configure_tensorflow(tf):
    """Konfigurasi TensorFlow setelah benar-benar diperlukan."""
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass


def cleanup_tensorflow(tf=None, objects=None):
    """Lepaskan model dan objek TensorFlow setelah inference."""
    if objects:
        for obj_name in objects:
            try:
                del obj_name
            except Exception:
                pass

    if tf is not None:
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass

    gc.collect()


# ============================================================================
# LSTM - LOAD ONLY WHEN USER CLICKS
# ============================================================================

def predict_lstm(dap_start):
    dap_start = int(dap_start)

    # 21 observasi = 7 hari x 3 waktu.
    input_df = df[
        df["DAP"].between(dap_start, dap_start + 6)
    ].copy()

    input_df = input_df.sort_values(
        ["DAP", "_Time_Order"]
    )

    if len(input_df) != 21:
        raise ValueError(
            f"Input LSTM membutuhkan 21 observasi, "
            f"tetapi ditemukan {len(input_df)}."
        )

    features = [
        "DAP",
        "Soil Moisture (%)",
        "Temperature (°C)",
    ]

    X = (
        input_df[features]
        .astype(np.float32)
        .values
    )

    X = np.expand_dims(X, axis=0)

    # TensorFlow baru di-import ketika LSTM benar-benar digunakan.
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    configure_tensorflow(tf)

    model = None

    try:
        model = load_model(
            LSTM_MODEL_PATH,
            compile=False,
        )

        with open(LSTM_SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        pred_scaled = model.predict(
            X,
            verbose=0,
        )

        pred_scaled = np.asarray(
            pred_scaled
        ).reshape(-1, 1)

        predictions = scaler.inverse_transform(
            pred_scaled
        ).flatten()

    finally:
        # Model tidak disimpan di cache RAM.
        del X

        if model is not None:
            del model

        try:
            del scaler
        except Exception:
            pass

        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass

        gc.collect()

    # 7 hari berikutnya = 21 observasi.
    future_rows = []

    for day_offset in range(7):
        future_dap = dap_start + 7 + day_offset

        for waktu in ["08:00", "12:00", "16:00"]:
            future_rows.append(
                (future_dap, waktu)
            )

    result = pd.DataFrame({
        "Horizon": [
            f"H{i // 3 + 1}"
            for i in range(21)
        ],
        "DAP": [
            x[0]
            for x in future_rows
        ],
        "Time": [
            x[1]
            for x in future_rows
        ],
        "Prediksi Maturity (%)": np.round(
            predictions,
            2,
        ),
    })

    result["Fase"] = result[
        "Prediksi Maturity (%)"
    ].apply(get_phase)

    return result


# ============================================================================
# CNN - LOAD ONLY WHEN USER UPLOADS IMAGE
# ============================================================================

CLASS_NAMES = [
    "Fase_Vegetatif",
    "Fase_Pembentukan_Tajuk",
    "Fase_Siap_Panen",
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
        dtype=np.float32,
    ) / 255.0

    batch = np.expand_dims(
        array,
        axis=0,
    )

    import tensorflow as tf
    from tensorflow.keras.models import load_model

    configure_tensorflow(tf)

    # Layer custom CNN final yang serializable.
    @tf.keras.utils.register_keras_serializable(
        package="Pakcoy"
    )
    class ChannelAveragePool(tf.keras.layers.Layer):

        def call(self, inputs):
            return tf.reduce_mean(
                inputs,
                axis=-1,
                keepdims=True,
            )

    @tf.keras.utils.register_keras_serializable(
        package="Pakcoy"
    )
    class ChannelMaxPool(tf.keras.layers.Layer):

        def call(self, inputs):
            return tf.reduce_max(
                inputs,
                axis=-1,
                keepdims=True,
            )

    model = None

    try:
        model = load_model(
            CNN_MODEL_PATH,
            compile=False,
            custom_objects={
                "ChannelAveragePool": ChannelAveragePool,
                "ChannelMaxPool": ChannelMaxPool,
            },
        )

        probabilities = model.predict(
            batch,
            verbose=0,
        )[0]

    finally:
        del batch
        del array
        del image_resized

        if model is not None:
            del model

        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass

        gc.collect()

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
            2,
        ),
    })

    return (
        image,
        predicted_class,
        confidence,
        result,
    )


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("🌱 Monitoring Pakcoy")

st.sidebar.markdown(
    """
**Sistem Monitoring Pakcoy**

Monitoring berbasis dataset dengan:
- Dashboard monitoring
- Prediksi LSTM 7 hari
- Klasifikasi CNN
- Analytics / History
"""
)

st.sidebar.divider()

st.sidebar.metric(
    "Total Data",
    f"{len(df):,}",
)

st.sidebar.metric(
    "Rentang DAP",
    f"{int(df['DAP'].min())} - {int(df['DAP'].max())}",
)

st.sidebar.caption(
    "Model LSTM dan CNN dimuat hanya saat diperlukan "
    "untuk mengurangi penggunaan RAM."
)


# ============================================================================
# HEADER
# ============================================================================

st.title("🌱 SISTEM MONITORING PAKCOY")

st.caption(
    "Dashboard Monitoring • LSTM Prediction • "
    "CNN Classification • Analytics / History"
)


# ============================================================================
# TABS
# ============================================================================

tab_dashboard, tab_lstm, tab_cnn, tab_history = st.tabs(
    [
        "🌱 Dashboard",
        "🧠 LSTM Prediction",
        "📷 CNN Classification",
        "📊 Analytics / History",
    ]
)


# ============================================================================
# 1. DASHBOARD
# ============================================================================

with tab_dashboard:

    st.header("🌱 Dashboard Monitoring")

    st.write(
        "Replay seluruh 1.140 data dataset secara berurutan "
        "dari data pertama sampai data terakhir."
    )

    # Session state.
    if "dashboard_index" not in st.session_state:
        st.session_state.dashboard_index = 0

    if "dashboard_last_update" not in st.session_state:
        st.session_state.dashboard_last_update = time.time()

    if "dashboard_running" not in st.session_state:
        st.session_state.dashboard_running = True

    # Pengaturan autoplay.
    col_a, col_b, col_c = st.columns([1, 1, 2])

    with col_a:
        if st.button(
            "▶️ Mulai",
            use_container_width=True,
        ):
            st.session_state.dashboard_running = True
            st.session_state.dashboard_last_update = time.time()

    with col_b:
        if st.button(
            "⏸️ Pause",
            use_container_width=True,
        ):
            st.session_state.dashboard_running = False

    with col_c:
        speed = st.selectbox(
            "Interval autoplay",
            [1, 2, 3],
            index=0,
            format_func=lambda x: f"{x} detik / data",
            label_visibility="visible",
        )

    # Fragment hanya menjalankan bagian Dashboard.
    # Streamlit 1.45.1 mendukung st.fragment.
    @st.fragment(run_every=1)
    def dashboard_live():

        now = time.time()

        if (
            st.session_state.dashboard_running
            and (
                now
                - st.session_state.dashboard_last_update
                >= speed
            )
        ):
            st.session_state.dashboard_index = (
                st.session_state.dashboard_index + 1
            ) % len(df)

            st.session_state.dashboard_last_update = now

        index = int(
            st.session_state.dashboard_index
        )

        row = df.iloc[index]

        # Slider manual.
        selected_index = st.slider(
            "Replay Dataset",
            min_value=0,
            max_value=len(df) - 1,
            value=index,
            key="dashboard_replay_slider",
            format="%d",
        )

        if selected_index != index:
            st.session_state.dashboard_index = int(
                selected_index
            )
            st.session_state.dashboard_last_update = time.time()
            index = int(selected_index)
            row = df.iloc[index]

        phase = get_phase(
            row["Ground Truth Maturity Level (%)"]
        )

        # Metrics.
        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Data",
            f"{index + 1:,} / {len(df):,}",
        )

        c2.metric(
            "Day",
            int(row["Day"]),
        )

        c3.metric(
            "DAP / HST",
            int(row["DAP"]),
        )

        c4.metric(
            "Soil Moisture",
            f'{row["Soil Moisture (%)"]:.0f}%',
        )

        c5.metric(
            "Temperature",
            f'{row["Temperature (°C)"]:.1f}°C',
        )

        st.subheader("Informasi Data Saat Ini")

        info_df = pd.DataFrame({
            "Parameter": [
                "Tanggal",
                "Waktu",
                "Soil Condition",
                "Ground Truth Maturity",
                "Fase",
                "Ground Truth Criteria",
            ],
            "Nilai": [
                str(row["Timestamp"])[:10],
                str(row["Time"]),
                row["Soil Condition"],
                f'{row["Ground Truth Maturity Level (%)"]:.2f}%',
                phase,
                row["Ground Truth Criteria"],
            ],
        })

        st.dataframe(
            info_df,
            use_container_width=True,
            hide_index=True,
        )

        # History sampai data saat ini.
        history = df.iloc[:index + 1][
            [
                "Replay_Index",
                "Soil Moisture (%)",
                "Temperature (°C)",
                "Ground Truth Maturity Level (%)",
            ]
        ].set_index("Replay_Index")

        st.subheader("📈 Grafik Monitoring")

        st.line_chart(
            history,
            height=420,
        )

        st.caption(
            f"Replay aktif: Data {index + 1:,} dari {len(df):,}"
            if st.session_state.dashboard_running
            else f"Replay dijeda pada Data {index + 1:,}."
        )

    dashboard_live()


# ============================================================================
# 2. LSTM
# ============================================================================

with tab_lstm:

    st.header("🧠 Prediksi LSTM 7 Hari")

    st.write(
        "Model menggunakan 21 observasi input "
        "(7 hari × 3 waktu) untuk menghasilkan "
        "21 prediksi observasi berikutnya."
    )

    min_dap = int(df["DAP"].min())

    # Input harus tersedia untuk 7 hari.
    max_input_dap = int(
        df["DAP"].max() - 6
    )

    dap_start = st.number_input(
        "DAP Awal Input",
        min_value=min_dap,
        max_value=max_input_dap,
        value=min_dap,
        step=1,
    )

    if st.button(
        "🚀 Proses Prediksi LSTM",
        type="primary",
        key="btn_lstm",
    ):

        try:

            with st.spinner(
                "Memuat model LSTM dan melakukan prediksi..."
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
                len(result),
            )

            c2.metric(
                "Prediksi Minimum",
                f'{result["Prediksi Maturity (%)"].min():.2f}%',
            )

            c3.metric(
                "Prediksi Maksimum",
                f'{result["Prediksi Maturity (%)"].max():.2f}%',
            )

            st.subheader(
                "Tabel Prediksi 7 Hari"
            )

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True,
            )

            chart_data = result[
                ["DAP", "Prediksi Maturity (%)"]
            ].copy()

            chart_data["Observasi"] = np.arange(
                1,
                len(chart_data) + 1,
            )

            chart_data = chart_data.set_index(
                "Observasi"
            )[["Prediksi Maturity (%)"]]

            st.subheader(
                "📈 Grafik Prediksi Maturity"
            )

            st.line_chart(
                chart_data,
                height=400,
            )

        except Exception as e:
            st.error(
                f"Terjadi error LSTM: {e}"
            )


# ============================================================================
# 3. CNN
# ============================================================================

with tab_cnn:

    st.header("📷 CNN Classification")

    st.write(
        "Upload foto tanaman Pakcoy untuk "
        "mengklasifikasikan fase pertumbuhan."
    )

    st.info(
        "Kelas CNN: Fase Vegetatif, "
        "Fase Pembentukan Tajuk, dan Fase Siap Panen."
    )

    uploaded_file = st.file_uploader(
        "📤 Upload foto Pakcoy",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        key="cnn_uploader",
    )

    if uploaded_file is not None:

        try:

            with st.spinner(
                "Memuat model CNN dan melakukan klasifikasi..."
            ):
                (
                    image,
                    predicted_class,
                    confidence,
                    result,
                ) = predict_cnn(
                    uploaded_file
                )

            c1, c2 = st.columns(
                [1, 1]
            )

            with c1:

                st.image(
                    image,
                    caption="Foto Pakcoy",
                    use_container_width=True,
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
                    f"{confidence * 100:.2f}%",
                )

                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True,
                )

            st.subheader(
                "📊 Grafik Confidence"
            )

            confidence_chart = result.set_index(
                "Kelas"
            )[["Confidence (%)"]]

            st.bar_chart(
                confidence_chart,
                height=350,
            )

        except Exception as e:
            st.error(
                f"Terjadi error CNN: {e}"
            )


# ============================================================================
# 4. ANALYTICS / HISTORY
# ============================================================================

with tab_history:

    st.header("📊 Analytics / History")

    st.write(
        "Riwayat seluruh data monitoring Pakcoy."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Data",
        f"{len(df):,}",
    )

    c2.metric(
        "DAP Minimum",
        int(df["DAP"].min()),
    )

    c3.metric(
        "DAP Maksimum",
        int(df["DAP"].max()),
    )

    c4.metric(
        "Jumlah Hari",
        int(df["Day"].nunique()),
    )

    st.subheader("Filter DAP")

    dap_filter = st.slider(
        "Pilih DAP",
        min_value=int(df["DAP"].min()),
        max_value=int(df["DAP"].max()),
        value=int(df["DAP"].min()),
        key="history_dap",
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
            errors="ignore",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Statistik Dataset")

    stat_df = pd.DataFrame({
        "Parameter": [
            "Soil Moisture (%)",
            "Temperature (°C)",
            "Ground Truth Maturity (%)",
        ],
        "Minimum": [
            df["Soil Moisture (%)"].min(),
            df["Temperature (°C)"].min(),
            df["Ground Truth Maturity Level (%)"].min(),
        ],
        "Rata-rata": [
            df["Soil Moisture (%)"].mean(),
            df["Temperature (°C)"].mean(),
            df["Ground Truth Maturity Level (%)"].mean(),
        ],
        "Maksimum": [
            df["Soil Moisture (%)"].max(),
            df["Temperature (°C)"].max(),
            df["Ground Truth Maturity Level (%)"].max(),
        ],
    })

    st.dataframe(
        stat_df.round(2),
        use_container_width=True,
        hide_index=True,
    )

    # Grafik history dibuat menggunakan chart native Streamlit,
    # bukan matplotlib, agar tidak menumpuk object figure di RAM.

    st.subheader("📈 History Ground Truth Maturity")

    maturity_history = (
        df.groupby("DAP", as_index=True)[
            "Ground Truth Maturity Level (%)"
        ]
        .mean()
        .to_frame()
    )

    st.line_chart(
        maturity_history,
        height=350,
    )

    st.subheader("💧 History Soil Moisture")

    moisture_history = (
        df.groupby("DAP", as_index=True)[
            "Soil Moisture (%)"
        ]
        .mean()
        .to_frame()
    )

    st.line_chart(
        moisture_history,
        height=350,
    )

    st.subheader("🌡️ History Temperature")

    temperature_history = (
        df.groupby("DAP", as_index=True)[
            "Temperature (°C)"
        ]
        .mean()
        .to_frame()
    )

    st.line_chart(
        temperature_history,
        height=350,
    )


# ============================================================================
# FOOTER
# ============================================================================

st.divider()

st.caption(
    "Monitoring Pakcoy MILLI | "
    "Dashboard + LSTM + CNN + Analytics / History"
)
