import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODEL_CONFIG = {
    "Decision Tree": {
        "slug": "decision_tree",
        "description": "Decision Tree Classifier dari implementasi awal project.",
    },
    "KNN": {
        "slug": "knn",
        "description": "K-Nearest Neighbors mengikuti notebook KNN.",
    },
    "Neural Network": {
        "slug": "nn",
        "description": "MLPClassifier mengikuti notebook Neural Network.",
    },
    "SVM": {
        "slug": "svm",
        "description": "Support Vector Machine mengikuti notebook SVM.",
    },
}

METHOD_CONFIG = {
    "Non-HPO": {
        "slug": "non_hpo",
        "description": "Model tanpa pencarian hyperparameter.",
    },
    "HPO": {
        "slug": "hpo",
        "description": "Model dengan hyperparameter optimization.",
    },
}


def artifact_paths(model_slug, method_slug):
    return {
        "model": BASE_DIR / "models" / model_slug / f"{method_slug}.pkl",
        "metrics": BASE_DIR / "outputs" / model_slug / method_slug / "metrics.json",
        "confusion_matrix": BASE_DIR / "outputs" / model_slug / method_slug / "confusion_matrix.png",
        "feature_importance": BASE_DIR / "outputs" / model_slug / method_slug / "feature_importance.png",
    }


@st.cache_resource
def load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        st.error(f"Model belum ditemukan: {path}")
        st.info("Jalankan training dulu dengan: python src/train_all.py")
        st.stop()
    return joblib.load(path)


@st.cache_data
def load_json(path: str):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def prediction_probabilities(model, input_data):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        yes_index = classes.index(1) if 1 in classes else len(classes) - 1
        prob_yes = float(probabilities[yes_index])
        return prob_yes, 1.0 - prob_yes

    if hasattr(model, "decision_function"):
        score = float(np.ravel(model.decision_function(input_data))[0])
        prob_yes = float(1 / (1 + np.exp(-score)))
        return prob_yes, 1.0 - prob_yes

    prediction = int(model.predict(input_data)[0])
    return float(prediction), float(1 - prediction)


def metrics_table():
    rows = []
    for model_name, model_config in MODEL_CONFIG.items():
        for method_name, method_config in METHOD_CONFIG.items():
            paths = artifact_paths(model_config["slug"], method_config["slug"])
            metrics = load_json(str(paths["metrics"]))
            if not metrics:
                continue
            rows.append(
                {
                    "Model": model_name,
                    "Metode": method_name,
                    "Accuracy": metrics.get("accuracy"),
                    "Precision Yes": metrics.get("precision_yes"),
                    "Recall Yes": metrics.get("recall_yes"),
                    "F1 Yes": metrics.get("f1_yes"),
                    "ROC-AUC": metrics.get("roc_auc"),
                }
            )
    return pd.DataFrame(rows)


st.set_page_config(
    page_title="Prediksi Campaign Marketing Bank",
    page_icon="B",
    layout="wide",
)

st.title("Prediksi Keberhasilan Campaign Marketing Bank")
st.write(
    "Aplikasi ini membandingkan empat model klasifikasi: KNN, Neural Network, "
    "Decision Tree, dan SVM. Setiap model memiliki versi Non-HPO dan HPO."
)

st.warning(
    "Catatan: fitur duration digunakan sebagai bagian dari data interaksi campaign. "
    "Jadi framing yang tepat adalah prediksi berdasarkan data nasabah dan interaksi campaign, "
    "bukan prediksi murni sebelum nasabah ditelepon."
)

with st.sidebar.form("model_selection_form"):
    options = []
    for m in MODEL_CONFIG.keys():
        for method in METHOD_CONFIG.keys():
            options.append(f"{m} - {method}")
    selected_option = st.selectbox("Pilih Konfigurasi Model", options)
    submitted = st.form_submit_button("Tampilkan Hasil", type="primary")

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Decision Tree"
    st.session_state.selected_method = "Non-HPO"

if submitted:
    model_part, method_part = selected_option.split(" - ")
    st.session_state.selected_model = model_part
    st.session_state.selected_method = method_part

selected_model = st.session_state.selected_model
selected_method = st.session_state.selected_method

model_slug = MODEL_CONFIG[selected_model]["slug"]
method_slug = METHOD_CONFIG[selected_method]["slug"]
paths = artifact_paths(model_slug, method_slug)

model = load_model(str(paths["model"]))
metrics = load_json(str(paths["metrics"]))

st.sidebar.info(MODEL_CONFIG[selected_model]["description"])
st.sidebar.caption(METHOD_CONFIG[selected_method]["description"])
if metrics:
    st.sidebar.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    st.sidebar.metric("F1 Yes", f"{metrics.get('f1_yes', 0):.2%}")

main_tab, compare_tab, eval_tab, about_tab = st.tabs(
    ["Prediksi", "Perbandingan", "Evaluasi Model", "Tentang Project"]
)

with main_tab:
    st.header(f"Form Prediksi - {selected_model} {selected_method}")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Umur", min_value=18, max_value=100, value=35)
        job = st.selectbox(
            "Pekerjaan",
            [
                "admin.",
                "blue-collar",
                "entrepreneur",
                "housemaid",
                "management",
                "retired",
                "self-employed",
                "services",
                "student",
                "technician",
                "unemployed",
                "unknown",
            ],
        )
        marital = st.selectbox("Status Pernikahan", ["married", "single", "divorced"])
        education = st.selectbox("Pendidikan", ["primary", "secondary", "tertiary", "unknown"])
        default = st.selectbox("Memiliki Kredit Default?", ["no", "yes"])
        balance = st.number_input("Saldo Rekening", value=1000)

    with col2:
        housing = st.selectbox("Memiliki Kredit Rumah?", ["no", "yes"])
        loan = st.selectbox("Memiliki Pinjaman Pribadi?", ["no", "yes"])
        contact = st.selectbox("Jenis Kontak", ["cellular", "telephone", "unknown"])
        day = st.number_input("Tanggal Kontak", min_value=1, max_value=31, value=15)
        month = st.selectbox(
            "Bulan Kontak",
            ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        )

    with col3:
        duration = st.number_input(
            "Durasi Panggilan (detik)",
            min_value=0,
            max_value=5000,
            value=300,
            help="Fitur kuat untuk akurasi, tetapi baru diketahui saat atau setelah campaign berlangsung.",
        )
        campaign = st.number_input("Jumlah Kontak pada Campaign Ini", min_value=1, max_value=50, value=1)
        pdays = st.number_input(
            "Hari Sejak Campaign Sebelumnya (-1 jika belum pernah)",
            min_value=-1,
            max_value=999,
            value=-1,
        )
        previous = st.number_input("Jumlah Kontak Campaign Sebelumnya", min_value=0, max_value=50, value=0)
        poutcome = st.selectbox("Hasil Campaign Sebelumnya", ["failure", "other", "success", "unknown"])

    input_data = pd.DataFrame(
        [
            {
                "age": age,
                "job": job,
                "marital": marital,
                "education": education,
                "default": default,
                "balance": balance,
                "housing": housing,
                "loan": loan,
                "contact": contact,
                "day": day,
                "month": month,
                "duration": duration,
                "campaign": campaign,
                "pdays": pdays,
                "previous": previous,
                "poutcome": poutcome,
            }
        ]
    )

    st.subheader("Preview Input")
    st.dataframe(input_data, width="stretch")

    if st.button("Prediksi", type="primary"):
        prediction = int(model.predict(input_data)[0])
        prob_yes, prob_no = prediction_probabilities(model, input_data)

        st.subheader("Hasil Prediksi")
        result_col1, result_col2, result_col3 = st.columns(3)

        if prediction == 1:
            result_col1.success("Berpotensi Deposit")
        else:
            result_col1.warning("Tidak Berpotensi Deposit")

        result_col2.metric("Probabilitas Deposit", f"{prob_yes:.2%}")
        result_col3.metric("Probabilitas Tidak Deposit", f"{prob_no:.2%}")

with compare_tab:
    st.header("Perbandingan Semua Model")
    comparison = metrics_table()
    if comparison.empty:
        st.info("Belum ada metrics yang tersedia. Jalankan training dulu.")
    else:
        formatted = comparison.copy()
        for column in ["Accuracy", "Precision Yes", "Recall Yes", "F1 Yes", "ROC-AUC"]:
            formatted[column] = formatted[column].map(lambda value: f"{value:.2%}")
        st.dataframe(formatted, width="stretch", hide_index=True)

with eval_tab:
    st.header(f"Evaluasi - {selected_model} {selected_method}")

    if metrics:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
        col2.metric("Precision Yes", f"{metrics.get('precision_yes', 0):.2%}")
        col3.metric("Recall Yes", f"{metrics.get('recall_yes', 0):.2%}")
        col4.metric("F1 Yes", f"{metrics.get('f1_yes', 0):.2%}")
        col5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.2%}")

        if selected_method == "HPO" and "best_params" in metrics:
            st.subheader("Best Parameter HPO")
            st.json(metrics["best_params"])
    else:
        st.info("File metrics belum tersedia.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        if paths["confusion_matrix"].exists():
            st.image(Image.open(paths["confusion_matrix"]), width="stretch")
        else:
            st.info("Confusion matrix belum tersedia.")

    with col2:
        st.subheader("Feature Importance")
        if paths["feature_importance"].exists():
            st.image(Image.open(paths["feature_importance"]), width="stretch")
        else:
            st.info("Feature importance hanya tersedia untuk model yang mendukung.")

with about_tab:
    st.header("Tentang Project")
    st.write(
        """
        **Topik:** Prediksi Keberhasilan Campaign Marketing Bank terhadap Nasabah

        **Jenis Masalah:** Klasifikasi biner

        **Target:** `deposit` dengan kelas `yes` dan `no`

        **Model:** KNN, Neural Network, Decision Tree, dan SVM

        **Perbandingan:** Non-HPO vs HPO
        """
    )

    st.subheader("Struktur Artefak")
    st.write(
        """
        Setiap model punya folder sendiri di `models/` dan `outputs/`.
        Struktur ini dibuat agar notebook, model, metrics, dan gambar evaluasi tidak bercampur.
        """
    )
