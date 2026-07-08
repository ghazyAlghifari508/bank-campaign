import json
import sys

def update_svm(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Insert a new cell after "Cek missing values"
    insert_index = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and "Cek missing values" in "".join(cell.get('source', [])):
            insert_index = i
            break
    
    if insert_index != -1:
        # Check if already inserted
        already_has_outlier = False
        for cell in nb['cells']:
            if "outlier_cols" in "".join(cell.get('source', [])):
                already_has_outlier = True
                break
                
        if not already_has_outlier:
            outlier_cell = {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Capping Outliers menggunakan metode IQR\n",
                    "outlier_cols = ['balance', 'duration', 'campaign', 'pdays', 'previous']\n",
                    "for col in outlier_cols:\n",
                    "    Q1 = df[col].quantile(0.25)\n",
                    "    Q3 = df[col].quantile(0.75)\n",
                    "    IQR = Q3 - Q1\n",
                    "    lower_bound = Q1 - 1.5 * IQR\n",
                    "    upper_bound = Q3 + 1.5 * IQR\n",
                    "    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)\n",
                    "\n",
                    "print(\"Outliers berhasil di-cap menggunakan nilai batas IQR.\")\n"
                ],
                "execution_count": None,
                "outputs": []
            }
            nb['cells'].insert(insert_index + 1, outlier_cell)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    update_svm(sys.argv[1])
