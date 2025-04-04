import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset dari Google Colab (Pastikan sudah diunggah)
file_path = "/content/train.csv"  # Sesuaikan path jika file ada di Drive
df = pd.read_csv(file_path)

# 1. Menampilkan informasi dataset
print("\n📌 Informasi Dataset:")
df.info()

# 2. Menampilkan 5 baris pertama dataset
print("\n📌 Lima Baris Pertama Dataset:")
print(df.head())

# 3. Statistik deskriptif
print("\n📌 Statistik Deskriptif:")
print(df.describe())

# 4. Menampilkan Q1 (25%), Q2 (50% atau median), Q3 (75%)
print("\n📌 Kuartil (Q1, Median/Q2, Q3) untuk setiap fitur numerik:")
print(df.describe(percentiles=[0.25, 0.5, 0.75]))

# 5. Menampilkan jumlah missing values
print("\n📌 Jumlah Missing Values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# 6. Visualisasi missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("🔍 Visualisasi Missing Values dalam Dataset", fontsize=14)
plt.show()

# 7. Histogram distribusi data numerik
df.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle("📊 Histogram Distribusi Fitur Numerik", fontsize=16)
plt.show()
