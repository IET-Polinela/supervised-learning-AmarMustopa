from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(X_train, X_test, y_train, y_test, title):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"📊 {title}")
    print(f"✅ MSE: {mse:.2f}")
    print(f"✅ R²: {r2:.4f}")

    # Scatter Plot Pred vs Actual
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.title(f"{title} - Scatter Plot")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()

    # Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(6,4))
    sns.histplot(residuals, kde=True)
    plt.title(f"{title} - Residual Distribution")
    plt.xlabel("Residual")
    plt.show()

    return mse, r2

# Data dengan outlier (pakai data encoded dari soal 2)
mse1, r2_1 = evaluate_model(X_train, X_test, y_train, y_test, "Linear Regression (With Outliers)")

# Data tanpa outlier + StandardScaler
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_std, y_no_out, test_size=0.2, random_state=42)
mse2, r2_2 = evaluate_model(X_train2, X_test2, y_train2, y_test2, "Linear Regression (Cleaned + Scaled)")
