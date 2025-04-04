import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Boxplot semua fitur numerik
plt.figure(figsize=(20, 12))
df_num = df.select_dtypes(include=[np.number])
df_num.boxplot(rot=90)
plt.title("Boxplot semua fitur numerik")
plt.show()

# Deteksi outlier dengan IQR
Q1 = df_num.quantile(0.25)
Q3 = df_num.quantile(0.75)
IQR = Q3 - Q1
is_outlier = ((df_num < (Q1 - 1.5 * IQR)) | (df_num > (Q3 + 1.5 * IQR)))

# Buat dataset tanpa outlier
df_no_outlier = df[~is_outlier.any(axis=1)]
print(f"✅ Dataset awal: {df.shape[0]} baris")
print(f"✅ Dataset tanpa outlier: {df_no_outlier.shape[0]} baris")
