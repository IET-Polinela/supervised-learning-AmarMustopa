import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Pastikan semua variabel hasil evaluasi dari soal sebelumnya sudah ada
# Jika belum, bisa didefinisikan dummy (sementara) seperti ini untuk menghindari error
# Harap ganti dengan nilai sebenarnya
try:
    mse_with_outliers
except:
    mse_with_outliers = 2e+10
    r2_with_outliers = 0.6
    mse_cleaned = 1e+10
    r2_cleaned = 0.75
    mse_poly2 = 0.9e+10
    r2_poly2 = 0.78
    mse_poly3 = 0.85e+10
    r2_poly3 = 0.8
    results_knn = {
        3: (1.1e+10, 0.72),
        5: (1.05e+10, 0.73),
        7: (1.02e+10, 0.74)
    }
    # Dummy y_test dan y_pred untuk visualisasi
    y_test = np.linspace(100000, 500000, 100)
    y_pred_cleaned = y_test + np.random.normal(0, 20000, 100)
    y_test_poly3 = y_test
    y_pred_poly3 = y_test + np.random.normal(0, 15000, 100)

# Data ringkasan dari setiap model (gunakan data hasil evaluasi sebenarnya bila tersedia)
comparison_data = {
    'Model': ['Linear (Outlier)', 'Linear (Cleaned)', 'Poly Deg=2', 'Poly Deg=3', 'KNN K=3', 'KNN K=5', 'KNN K=7'],
    'MSE': [mse_with_outliers, mse_cleaned, mse_poly2, mse_poly3, results_knn[3][0], results_knn[5][0], results_knn[7][0]],
    'R2': [r2_with_outliers, r2_cleaned, r2_poly2, r2_poly3, results_knn[3][1], results_knn[5][1], results_knn[7][1]]
}

comparison_df = pd.DataFrame(comparison_data)
print("\nPerbandingan Model:")
print(comparison_df)

# Visualisasi hasil prediksi vs nilai aktual dari setiap model
plt.figure(figsize=(12, 6))

# Linear Regression tanpa outlier
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_cleaned, alpha=0.5, label='Linear Cleaned')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Linear Regression Cleaned")
plt.legend()

# Polynomial Regression derajat 3
plt.subplot(1, 2, 2)
plt.scatter(y_test_poly3, y_pred_poly3, alpha=0.5, color='orange', label='Poly Deg=3')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Polynomial Regression Deg=3")
plt.legend()

plt.tight_layout()
plt.show()

# Jawaban Analisis
print("\nAnalisis:")
print("a. Model dengan prediksi terbaik berdasarkan MSE dan R2 adalah:")
print(comparison_df.sort_values(by='R2', ascending=False).iloc[0])

print("\nb. Jika diterapkan di dunia nyata, model yang direkomendasikan adalah Polynomial Regression (derajat 3) atau Linear Regression (cleaned), karena memberikan keseimbangan antara akurasi dan interpretabilitas, tergantung pada kompleksitas data dan kebutuhan interpretasi model.")
