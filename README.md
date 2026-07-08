# Prediksi Campaign Marketing Bank (Tugas Besar)

🌐 **Live Website (Streamlit):** [https://bank-campaign-tubes.streamlit.app/](https://bank-campaign-tubes.streamlit.app/)

Project ini berisi implementasi dan perbandingan empat algoritma *Machine Learning* untuk memprediksi keberhasilan *campaign* pemasaran (telemarketing) pada nasabah bank, apakah mereka akan membuka deposito berjangka atau tidak.

### Algoritma yang Dieksplorasi:
1. **K-Nearest Neighbors (KNN)**
2. **Decision Tree**
3. **Support Vector Machine (SVM)**
4. **Neural Network (NN)**

Setiap model dikembangkan dalam dua versi:
- **Non-HPO**: Model dasar (*baseline*) dengan parameter *default*.
- **HPO**: Model yang telah dioptimasi menggunakan *Hyperparameter Optimization* (GridSearchCV).

Target prediksi pada dataset ini adalah kolom `deposit`:
- `yes`: *Campaign* berhasil, nasabah mengambil deposito.
- `no`: *Campaign* tidak berhasil.

---

## Metodologi Utama
- **Penanganan Outlier**: Menggunakan metode batas IQR (Interquartile Range) pada fitur numerik.
- **Pembagian Data**: Dataset dibagi secara seragam menggunakan `train_test_split` dengan rasio **80% data latih** dan **20% data uji** (`random_state=42`, `stratify=y`).
- **Feature Scaling**: Menggunakan `StandardScaler` (diterapkan sesuai dengan karakteristik masing-masing algoritma).

## 🏆 Hasil Terbaik (Best Model)
Berdasarkan hasil pengujian, **Neural Network (dengan HPO)** merupakan model paling optimal dengan performa:
- **Akurasi**: 84.95%
- **F1-Score**: 84.59%
- **ROC-AUC**: 0.9185

---

## Catatan Penting (Framing Presentasi)
Project ini menggunakan fitur `duration` (durasi telepon). Framing presentasi yang benar kepada dosen/penguji:
> *"Model memprediksi keberhasilan campaign berdasarkan data profil nasabah **dan interaksi campaign yang telah berlangsung** (termasuk durasi panggilan)."*

Jangan mengklaim bahwa model ini murni untuk prediksi *sebelum* nasabah ditelepon, karena `duration` baru diketahui setelah panggilan selesai.

---

## Struktur Folder

```text
bank-campaign/
|-- app.py                  # Aplikasi Streamlit (Web UI)
|-- requirements.txt        # Daftar dependency Python
|-- data/
|   |-- bank.csv            # Dataset utama
|-- notebooks/              # Lembar kerja Jupyter Notebook per model
|-- src/                    # Source code pipeline training model
|   |-- train_all.py        # Skrip utama untuk menjalankan training seluruh model
|-- models/                 # File model (.pkl) hasil training
|-- outputs/                # Hasil evaluasi (metrics.json, confusion matrix, feature importance)
```

## Cara Menjalankan Project

**1. Masuk ke folder project:**
```powershell
cd "C:\Coding\Data Science\Tugas_Besar\bank-campaign"
```

**2. Aktifkan virtual environment & Install dependency:**
```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

**3. (Opsional) Training ulang seluruh model:**
```powershell
python src/train_all.py
```

**4. Jalankan aplikasi Web UI:**
```powershell
streamlit run app.py
```
