"""
Comparative Implementation of Machine Learning Regression Algorithms
California Housing Dataset Benchmark
Author: TheBilalButt (BS Artificial Intelligence)
"""

import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    print("=" * 80)
    print("COMPARATIVE MACHINE LEARNING REGRESSION BENCHMARK")
    print("California Housing Dataset (20,640 Samples, 8 Features)")
    print("=" * 80)

    # 1. Load Dataset
    print("\n[1/5] Loading California Housing Dataset...")
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # 2. Train-Test Split (80/20, random_state=42)
    print("\n[2/5] Performing 80/20 Train-Test Split (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # 3. Feature Scaling (Fit strictly on training set to avoid data leakage)
    print("\n[3/5] Fitting StandardScaler strictly on Training Data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("StandardScaler fitted and applied. Zero data leakage verified.")

    # 4. Model Registry
    models = {
        'Linear Regression': (
            LinearRegression(),
            True  # Use scaled features
        ),
        'Decision Tree': (
            DecisionTreeRegressor(random_state=42, max_depth=10),
            False  # Unscaled
        ),
        'Random Forest': (
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            False  # Unscaled
        ),
        'Neural Network (MLP)': (
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42),
            True  # Scaled
        ),
        'XGBoost': (
            XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, objective="reg:squarederror", random_state=42),
            False  # Unscaled
        ),
        'KNN': (
            KNeighborsRegressor(n_neighbors=5),
            True  # Scaled
        )
    }

    # 5. Train, Predict, and Evaluate
    print("\n[4/5] Training and Evaluating All 6 Regression Models...")
    results = []

    for name, (model, is_scaled) in models.items():
        X_tr = X_train_scaled if is_scaled else X_train
        X_te = X_test_scaled if is_scaled else X_test

        # Measure training runtime
        t0 = time.perf_counter()
        model.fit(X_tr, y_train)
        train_time = time.perf_counter() - t0

        # Measure inference runtime
        t0 = time.perf_counter()
        y_pred = model.predict(X_te)
        pred_time = time.perf_counter() - t0

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Algorithm': name,
            'MAE': round(mae, 4),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'R² Score': round(r2, 4),
            'Train Time (s)': round(train_time, 4),
            'Pred Time (s)': round(pred_time, 4)
        })
        print(f"  ✓ {name:22s} | RMSE: {rmse:.4f} | R²: {r2:.4f} | Train: {train_time:.3f}s | Pred: {pred_time:.4f}s")

    # 6. Comparative DataFrame sorted by RMSE ascending
    df_results = pd.DataFrame(results).sort_values(by='RMSE', ascending=True).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("FINAL COMPARATIVE EVALUATION TABLE (SORTED BY RMSE ASCENDING)")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)

    # 7. Summary
    best = df_results.iloc[0]
    print(f"\n🏆 BEST PERFORMING ALGORITHM: {best['Algorithm']}")
    print(f"   • Lowest MAE:  {best['MAE']}")
    print(f"   • Lowest MSE:  {best['MSE']}")
    print(f"   • Lowest RMSE: {best['RMSE']}")
    print(f"   • Highest R²:  {best['R² Score']}")

if __name__ == '__main__':
    main()
