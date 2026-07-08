import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []

# A. Biodata
cells.append(new_markdown_cell("""# A. Biodata Mahasiswa
## Mata Kuliah: Dasar Ilmu Data (GIK2GAB3)
## Materi : Model Decision Tree & Hyperparameter Tuning pada studi kasus Klasifikasi menggunakan dataset Bank Marketing

*   NIM  : 
*   Nama : 
*   Kelas: 

**Studi Kasus:** Memprediksi apakah seorang nasabah bank akan membuka rekening deposito berjangka (`deposit` = yes/no) setelah dihubungi melalui campaign marketing telepon.
"""))

# B. Load Library dan Dataset
cells.append(new_markdown_cell("# B. Load Library dan Dataset"))

cells.append(new_markdown_cell("## B1. Load library"))
cells.append(new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin"""))

cells.append(new_markdown_cell("## B2. Load dataset"))
cells.append(new_code_cell("""# Define path to dataset
BASE_DIR = Path().resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "bank.csv"

# Load dataset
df = pd.read_csv(DATA_PATH) 
print(f"Ukuran dataset: {df.shape[0]} baris, {df.shape[1]} kolom")
df.head()"""))

cells.append(new_markdown_cell("## B3. Preprocessing Data\nDataset ini memiliki kolom kategorikal yang perlu di-encode menjadi numerik. Kita juga menerapkan *Outlier Capping* menggunakan batas IQR pada data numerik."))

cells.append(new_code_cell("""class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns or ["balance", "duration", "campaign", "pdays", "previous"]

    def fit(self, X, y=None):
        self.bounds_ = {}
        for column in self.columns:
            q1 = X[column].quantile(0.25)
            q3 = X[column].quantile(0.75)
            iqr = q3 - q1
            self.bounds_[column] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        return self

    def transform(self, X):
        X = X.copy()
        for column, (lower, upper) in self.bounds_.items():
            X[column] = X[column].clip(lower=lower, upper=upper)
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features

def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(exclude="object").columns.tolist()
    categorical_features = X.select_dtypes(include="object").columns.tolist()

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return Pipeline(steps=[
        ("cap_outliers", OutlierCapper()),
        ("transform", transformer)
    ])"""))

# C. Klasifikasi
cells.append(new_markdown_cell("# C. Klasifikasi dengan Decision Tree"))
cells.append(new_markdown_cell("## C1. Tahap 1: Tentukan Fitur dan Label"))
cells.append(new_code_cell("""X = df.drop(columns=["deposit"])
y = (df["deposit"] == "yes").astype(int)

print(f'Jumlah fitur: {X.shape[1]}')
print(f'Jumlah data: {X.shape[0]}')"""))

cells.append(new_markdown_cell("## C2. Tahap 2: Bagi Dataset menjadi Data Latih (Train Data) dan Data Uji (Test Data)"))
cells.append(new_code_cell("""X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'Jumlah total data: {len(X)}')
print(f'Jumlah data latih: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)')
print(f'Jumlah data uji: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)')"""))

cells.append(new_markdown_cell("## C3. Tahap 3: Siapkan Classifier dan Tentukan Variabel/Parameternya"))
cells.append(new_code_cell("""classifier = Pipeline(
    steps=[
        ("preprocess", build_preprocessor(X)),
        ("model", DecisionTreeClassifier(random_state=42)),
    ]
)
print('Classifier Decision Tree (tanpa HPO):')
print(classifier)"""))

cells.append(new_markdown_cell("## C4. Tahap 4: Lakukan Proses Training dengan Data Latih"))
cells.append(new_code_cell("""import time
start_time = time.time()
classifier.fit(X_train, y_train)
training_time = time.time() - start_time

print(f'Training selesai!')
print(f'Waktu training: {training_time:.2f} detik')"""))


cells.append(new_markdown_cell("## C5. Tahap 5: Lakukan Pengujian dengan Data Uji"))
cells.append(new_code_cell("""y_pred = classifier.predict(X_test)

benar = (y_pred == y_test).sum()
salah = (y_pred != y_test).sum()
print(f'Jumlah prediksi benar: {benar} dari {len(y_test)} data')
print(f'Jumlah prediksi salah: {salah} dari {len(y_test)} data')"""))

cells.append(new_markdown_cell("## C6. Tahap 6: Analisa Performansi Model"))

cells.append(new_code_cell("""accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Score (tanpa HPO): {accuracy:.4f}')
print(f'Accuracy Percentage: {accuracy * 100:.2f}%')"""))

cells.append(new_code_cell("""cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix (tanpa HPO):')
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No (0)', 'Yes (1)'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix\\nTanpa HPO', fontsize=13, fontweight='bold')
plt.show()"""))

cells.append(new_code_cell("""report = classification_report(y_test, y_pred, target_names=['No (0)', 'Yes (1)'])
print('Classification Report (tanpa HPO):')
print(report)"""))

# E. HPO
cells.append(new_markdown_cell("# E. Tahap 2: Hyperparameter Optimization (HPO) dengan GridSearchCV"))
cells.append(new_markdown_cell("## E1. Siapkan Variabel Classifier dan Tentukan Parameter HPO"))
cells.append(new_code_cell("""param_grid = {
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": [6, 8, 10, 12],
    "model__min_samples_split": [2, 20],
    "model__min_samples_leaf": [5, 10, 20],
    "model__class_weight": [None, "balanced"],
}

hpo_pipeline = Pipeline(
    steps=[
        ("preprocess", build_preprocessor(X)),
        ("model", DecisionTreeClassifier(random_state=42)),
    ]
)

grid = GridSearchCV(
    estimator=hpo_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)"""))

cells.append(new_markdown_cell("## E2. Lakukan Proses Training HPO"))
cells.append(new_code_cell("""start_time = time.time()
grid.fit(X_train, y_train)
hpo_time = time.time() - start_time

print(f"Waktu training HPO: {hpo_time:.2f} detik")
print("Best parameters found: ", grid.best_params_)
hpo_model = grid.best_estimator_"""))


cells.append(new_markdown_cell("## E3. Pengujian dan Evaluasi Model HPO"))

cells.append(new_code_cell("""y_pred_hpo = hpo_model.predict(X_test)
accuracy_hpo = accuracy_score(y_test, y_pred_hpo)
print(f"Accuracy Score (dengan HPO): {accuracy_hpo:.4f}")
print(f"Accuracy Percentage: {accuracy_hpo * 100:.2f}%")"""))

cells.append(new_code_cell("""cm_hpo = confusion_matrix(y_test, y_pred_hpo)
print('Confusion Matrix (dengan HPO):')
print(cm_hpo)

disp_hpo = ConfusionMatrixDisplay(confusion_matrix=cm_hpo, display_labels=['No (0)', 'Yes (1)'])
disp_hpo.plot(cmap='Greens', values_format='d')
plt.title('Confusion Matrix\\nDengan HPO', fontsize=13, fontweight='bold')
plt.show()"""))

cells.append(new_code_cell("""report_hpo = classification_report(y_test, y_pred_hpo, target_names=['No (0)', 'Yes (1)'])
print('Classification Report (dengan HPO):')
print(report_hpo)"""))

# F. Kesimpulan
cells.append(new_markdown_cell("# F. Kesimpulan Umum Menggunakan HPO"))
cells.append(new_code_cell("""print("Perbandingan Akurasi Decision Tree:")
print(f"- Tanpa HPO: {accuracy:.4f}")
print(f"- Dengan HPO: {accuracy_hpo:.4f}")
print(f"Selisih Peningkatan: {(accuracy_hpo - accuracy):.4f}")
print(f"Parameter Terbaik HPO: {grid.best_params_}")"""))

nb['cells'] = cells

with open('notebooks/decision_tree/Decision_Tree_Model.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
