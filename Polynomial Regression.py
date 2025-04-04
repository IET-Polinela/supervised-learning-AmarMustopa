from sklearn.preprocessing import PolynomialFeatures

def polynomial_regression(degree):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X_std)

    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_poly, y_no_out, test_size=0.2, random_state=42)
    return evaluate_model(X_train_p, X_test_p, y_train_p, y_test_p, f"Polynomial Regression (Degree {degree})")

mse_deg2, r2_deg2 = polynomial_regression(2)
mse_deg3, r2_deg3 = polynomial_regression(3)
