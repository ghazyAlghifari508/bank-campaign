# Prediksi Campaign Marketing Bank

Project ini berisi aplikasi Streamlit untuk membandingkan empat model klasifikasi:

1. KNN
2. Neural Network
3. Decision Tree
4. SVM

Setiap model memiliki dua metode:

- Non-HPO: model tanpa hyperparameter optimization.
- HPO: model dengan hyperparameter optimization.

Target prediksi adalah kolom `deposit`:

- `yes`: campaign berhasil, nasabah mengambil deposito.
- `no`: campaign tidak berhasil.

## Catatan Penting

Project ini menggunakan fitur `duration`. Framing presentasi yang benar:

> Model memprediksi keberhasilan campaign berdasarkan data nasabah dan data interaksi campaign, termasuk durasi panggilan.

Jangan bilang model ini murni untuk prediksi sebelum nasabah ditelepon, karena `duration` baru diketahui saat atau setelah panggilan.

## Struktur Folder

```text
bank-campaign/
|-- app.py
|-- requirements.txt
|-- data/
|   |-- bank.csv
|-- notebooks/
|   |-- decision_tree/
|   |-- knn/
|   |-- nn/
|   |-- svm/
|-- src/
|   |-- common/
|   |-- decision_tree/
|   |-- knn/
|   |-- nn/
|   |-- svm/
|   |-- train_all.py
|-- models/
|   |-- decision_tree/
|   |-- knn/
|   |-- nn/
|   |-- svm/
|-- outputs/
|   |-- common/
|   |-- decision_tree/
|   |-- knn/
|   |-- nn/
|   |-- svm/
```

Di dalam `models/<nama_model>/` terdapat:

```text
non_hpo.pkl
hpo.pkl
```

Di dalam `outputs/<nama_model>/<metode>/` terdapat:

```text
metrics.json
confusion_matrix.png
feature_importance.png  # hanya untuk model yang mendukung
```

## Cara Menjalankan

Masuk ke folder project:

```powershell
cd "C:\Coding\Data Science\Tugas_Besar\bank-campaign"
```

Aktifkan virtual environment:

```powershell
.\.venv\Scripts\activate
```

Install dependency:

```powershell
pip install -r requirements.txt
```

Training semua model:

```powershell
python src/train_all.py
```

Jalankan aplikasi:

```powershell
streamlit run app.py
```

## Branch Kerja

Perubahan struktur empat model ini dikerjakan di branch:

```text
feature/compare-4-bank-models
```
