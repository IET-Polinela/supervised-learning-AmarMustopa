from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load ulang data (pastikan file csv tersedia di sesi runtime)
df = pd.read_csv('/content/train.csv')

# Handle missing values
# Drop kolom dengan missing value lebih dari 30%
threshold = 0.3 * len(df)
df = df.dropna(thresh=threshold, axis=1)

# Isi sisa missing values dengan median
df.fillna(df.median(numeric_only=True), inplace=True)

# Pisahkan fitur numerik
num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_features.remove('SalePrice')

X = df[num_features]
y = df['SalePrice']

# Hilangkan outlier berdasarkan IQR
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
X_no_out = X[mask]
y_no_out = y[mask]

# Feature scaling
scaler = StandardScaler()
X_std = scaler.fit_transform(X_no_out)

# Split data untuk KNN
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_std, y_no_out, test_size=0.2, random_state=42)

# Fungsi untuk training dan evaluasi model KNN dengan visualisasi
results_knn = {}
def knn_regression(k):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train_knn, y_train_knn)
    y_pred = model.predict(X_test_knn)

    mse = mean_squared_error(y_test_knn, y_pred)
    r2 = r2_score(y_test_knn, y_pred)
    results_knn[k] = (mse, r2)

    # Visualisasi
    plt.figure(figsize=(15, 4))

    # Scatter plot
    plt.subplot(1, 3, 1)
    plt.scatter(y_test_knn, y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"KNN (k={k}) - Actual vs Predicted")

    # Residual plot
    residuals = y_test_knn - y_pred
    plt.subplot(1, 3, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residual Plot")

    # Distribusi residual
    plt.subplot(1, 3, 3)
    plt.hist(residuals, bins=30, color='gray', edgecolor='black')
    plt.title("Distribution of Residuals")
    plt.tight_layout()
    plt.show()

    print(f"K={k} - MSE: {mse:.2f}, R2: {r2:.2f}\n")

# Evaluasi untuk k = 3, 5, 7
for k in [3, 5, 7]:
    knn_regression(k)

# Tampilkan hasil akhir dalam bentuk tabel
print("\nRingkasan KNN Regression:")
knn_summary = pd.DataFrame(results_knn, index=['MSE', 'R2']).T
print(knn_summary)
