import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_CONFIG = {
    "Non-HPO": {
        "model_path": BASE_DIR / "models" / "decision_tree_non_hpo.pkl",
        "metrics_path": BASE_DIR / "outputs" / "metrics_non_hpo.json",
        "cm_path": BASE_DIR / "outputs" / "confusion_matrix_non_hpo.png",
        "fi_path": BASE_DIR / "outputs" / "feature_importance_non_hpo.png",
        "description": "Decision Tree default tanpa pencarian hyperparameter.",
    },
    "HPO": {
        "model_path": BASE_DIR / "models" / "decision_tree_hpo.pkl",
        "metrics_path": BASE_DIR / "outputs" / "metrics_hpo.json",
        "cm_path": BASE_DIR / "outputs" / "confusion_matrix_hpo.png",
        "fi_path": BASE_DIR / "outputs" / "feature_importance_hpo.png",
        "description": "Decision Tree dengan Hyperparameter Optimization menggunakan GridSearchCV.",
    },
}

st.set_page_config(
    page_title="Prediksi Campaign Marketing Bank",
    page_icon="🏦",
    layout="wide",
)

@st.cache_resource
def load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        st.error("Model belum ditemukan. Jalankan dulu: python src/train_all.py")
        st.stop()
    return joblib.load(path)

@st.cache_data
def load_json(path: str):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

st.title("🏦 Prediksi Keberhasilan Campaign Marketing Bank terhadap Nasabah")
st.write(
    "Aplikasi ini membandingkan dua model **Decision Tree Classifier**, yaitu model "
    "**Non-HPO** dan model **HPO**. Target prediksi adalah apakah nasabah akan "
    "menerima produk deposito berjangka atau tidak."
)

st.warning(
    "Catatan jujur: aplikasi ini memakai fitur `duration` agar performa bisa masuk "
    "kisaran 80-an. Artinya, framing yang benar adalah prediksi berdasarkan data nasabah "
    "dan data interaksi campaign, bukan prediksi murni sebelum nasabah ditelepon."
)

selected_model = st.sidebar.selectbox("Pilih Jenis Model", list(MODEL_CONFIG.keys()))
config = MODEL_CONFIG[selected_model]
model = load_model(str(config["model_path"]))
metrics = load_json(str(config["metrics_path"]))

st.sidebar.info(config["description"])

if metrics:
    st.sidebar.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    st.sidebar.metric("F1 Yes", f"{metrics.get('f1_yes', 0):.2%}")

main_tab, eval_tab, about_tab = st.tabs(["Prediksi", "Evaluasi Model", "Tentang Project"])

with main_tab:
    st.header(f"Form Prediksi Nasabah - Model {selected_model}")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Umur", min_value=18, max_value=100, value=35)
        job = st.selectbox(
            "Pekerjaan",
            [
                "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                "retired", "self-employed", "services", "student", "technician",
                "unemployed", "unknown",
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
            help="Fitur kuat untuk akurasi, tetapi baru diketahui setelah/saat campaign berlangsung.",
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

    input_data = pd.DataFrame([
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
    ])

    st.subheader("Preview Input")
    st.dataframe(input_data, use_container_width=True)

    if st.button("Prediksi", type="primary"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        prob_no = probability[0]
        prob_yes = probability[1]

        st.subheader("Hasil Prediksi")
        result_col1, result_col2, result_col3 = st.columns(3)

        if prediction == 1:
            result_col1.success("Berpotensi Deposit")
        else:
            result_col1.warning("Tidak Berpotensi Deposit")

        result_col2.metric("Probabilitas Deposit", f"{prob_yes:.2%}")
        result_col3.metric("Probabilitas Tidak Deposit", f"{prob_no:.2%}")

with eval_tab:
    st.header(f"Evaluasi Model {selected_model}")

    if metrics:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
        col2.metric("Precision Yes", f"{metrics.get('precision_yes', 0):.2%}")
        col3.metric("Recall Yes", f"{metrics.get('recall_yes', 0):.2%}")
        col4.metric("F1 Yes", f"{metrics.get('f1_yes', 0):.2%}")
        col5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.2%}")

        if selected_model == "HPO" and "best_params" in metrics:
            st.subheader("Best Parameter HPO")
            st.json(metrics["best_params"])
    else:
        st.info("File metrics belum tersedia.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        if config["cm_path"].exists():
            st.image(Image.open(config["cm_path"]), use_container_width=True)
        else:
            st.info("Confusion matrix belum tersedia.")

    with col2:
        st.subheader("Feature Importance")
        if config["fi_path"].exists():
            st.image(Image.open(config["fi_path"]), use_container_width=True)
        else:
            st.info("Feature importance belum tersedia.")

with about_tab:
    st.header("Tentang Project")
    st.write(
        """
        **Topik:** Prediksi Keberhasilan Campaign Marketing Bank terhadap Nasabah  
        **Jenis Masalah:** Klasifikasi biner  
        **Target:** `deposit` dengan kelas `yes` dan `no`  
        **Model:** Decision Tree Classifier  
        **Perbandingan:** Non-HPO vs HPO
        """
    )

    st.subheader("Perbedaan Non-HPO dan HPO")
    st.write(
        """
        - **Non-HPO** memakai Decision Tree default tanpa proses pencarian parameter terbaik.
        - **HPO** memakai GridSearchCV untuk mencari kombinasi hyperparameter yang menghasilkan performa terbaik.
        """
    )

    st.subheader("Catatan Tentang Duration")
    st.write(
        """
        Fitur `duration` digunakan karena sangat berpengaruh terhadap keberhasilan campaign. 
        Namun, fitur ini baru diketahui ketika panggilan sudah dilakukan. Jadi, aplikasi ini lebih tepat 
        dijelaskan sebagai model prediksi berbasis data nasabah dan interaksi campaign, bukan model 
        targeting murni sebelum campaign dimulai.
        """
    )
