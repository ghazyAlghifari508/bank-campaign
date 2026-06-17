# Prediksi Keberhasilan Campaign Marketing Bank - Decision Tree

Project ini berisi aplikasi Streamlit untuk membandingkan dua model Decision Tree:

1. **Non-HPO**: Decision Tree default tanpa hyperparameter optimization.
2. **HPO**: Decision Tree dengan GridSearchCV.

Target prediksi adalah kolom `deposit`:

- `yes`: campaign berhasil, nasabah mengambil deposito.
- `no`: campaign tidak berhasil.

## Catatan Penting

Project ini menggunakan fitur `duration` agar performa model masuk kisaran 80-an. Jadi framing presentasi yang benar:

> Model memprediksi keberhasilan campaign berdasarkan data nasabah dan data interaksi campaign, termasuk durasi panggilan.

Jangan bilang model ini murni untuk prediksi sebelum nasabah ditelepon, karena `duration` baru diketahui saat/setelah panggilan.

## Hasil Training Saat Ini

| Model | Accuracy | Precision Yes | Recall Yes | F1 Yes | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Non-HPO | 0.7940 | 0.7892 | 0.7713 | 0.7801 | 0.7929 |
| HPO | 0.8249 | 0.7923 | 0.8544 | 0.8222 | 0.8893 |

Best parameter HPO:

```text
{
  "model__class_weight": "balanced",
  "model__criterion": "gini",
  "model__max_depth": 10,
  "model__min_samples_leaf": 5,
  "model__min_samples_split": 20
}
```

## Cara Menjalankan

Masuk ke folder project:

```powershell
cd "C:\Big Project\bank_campaign_dt_hpo_vs_non_hpo"
```

Buat virtual environment:

```powershell
python -m venv .venv
.venv\Scriptsctivate
```

Install dependency:

```powershell
pip install -r requirements.txt
```

Training ulang model:

```powershell
python src/train_all.py
```

Jalankan aplikasi:

```powershell
streamlit run app.py
```

## Struktur Folder

```text
bank_campaign_dt_hpo_vs_non_hpo/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── bank.csv
├── models/
│   ├── decision_tree_non_hpo.pkl
│   └── decision_tree_hpo.pkl
├── outputs/
│   ├── metrics_non_hpo.json
│   ├── metrics_hpo.json
│   ├── confusion_matrix_non_hpo.png
│   ├── confusion_matrix_hpo.png
│   ├── feature_importance_non_hpo.png
│   └── feature_importance_hpo.png
└── src/
    └── train_all.py
```
