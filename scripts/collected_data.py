import csv
from pathlib import Path
from sklearn.datasets import load_diabetes

# Load dataset
dataset = load_diabetes()

# Nama Fitur
feature_names = dataset.feature_names

# Data input (X) dan target (y)
X = dataset.data
y = dataset.target

# save file dataset ke folder "data"
data_folder = Path("data")
data_file = data_folder / "diabetes_regression.csv"

# Buka file CSV Dataset
with open(data_file, "w", newline="") as file:
    writer = csv.writer(file)

    header = list(feature_names) + ["target"]
    writer.writerow(header)

    for feature, target in zip(X,y):
        row = list(feature) + [target]
        writer.writerow(row)

# Menampilkan informasi hasil
print("Dataset berhasil tersimpan")
print(f" Lokasi file: {data_file}")
print(f" Jumlah data: {len(y)}")
print(f" Nam Fitur: {', '.join(feature_names)}")
