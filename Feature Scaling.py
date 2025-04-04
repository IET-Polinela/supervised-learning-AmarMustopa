from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Drop kategorikal lalu encode lagi
df_no_outlier_enc = pd.get_dummies(df_no_outlier)
X_no_out = df_no_outlier_enc.drop('SalePrice', axis=1)
y_no_out = df_no_outlier_enc['SalePrice']

# Scaling
scaler_std = StandardScaler()
scaler_minmax = MinMaxScaler()

X_std = scaler_std.fit_transform(X_no_out)
X_minmax = scaler_minmax.fit_transform(X_no_out)

# Plot histogram sebelum & sesudah
def plot_hist_scaled(data, title):
    plt.figure(figsize=(15, 5))
    plt.hist(data[:, 0], bins=50, alpha=0.7)
    plt.title(title)
    plt.show()

plot_hist_scaled(X_no_out.values, "Distribusi Sebelum Scaling")
plot_hist_scaled(X_std, "StandardScaler")
plot_hist_scaled(X_minmax, "MinMaxScaler")
