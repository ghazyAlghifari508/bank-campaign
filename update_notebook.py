import json
import sys

def update_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replacements for KNN
    content = content.replace(
        "Pembagian dataset: **70% data latih** dan **30% data uji**.",
        "Pembagian dataset: **80% data latih** dan **20% data uji**."
    )
    content = content.replace(
        "Bagi data dengan rasio 70% latih dan 30% uji",
        "Bagi data dengan rasio 80% latih dan 20% uji"
    )
    content = content.replace(
        "test_size=0.3",
        "test_size=0.2"
    )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    update_notebook(sys.argv[1])
