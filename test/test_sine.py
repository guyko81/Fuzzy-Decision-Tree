import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import sys
import os

# Add parent directory to path to import fuzzy_tree
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fuzzy_tree import FuzzyDecisionTreeRegressor
from fuzzy_boosted_tree import FuzzyBoostedTreeRegressor

def generate_sine_data(n_samples=1000, noise=0.1):
    """Generate sine wave data with noise"""
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y = np.sin(X.ravel()) + np.random.normal(0, noise, n_samples)
    return X, y

def plot_predictions(X, y, predictions, labels):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(15, 5))
    
    # Sort X and corresponding predictions for smooth plotting
    sort_idx = np.argsort(X.ravel())
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    # Plot true function
    plt.scatter(X, y, color='gray', alpha=0.5, label='Data points')
    plt.plot(X_sorted, np.sin(X_sorted), 'k--', label='True sine')
    
    # Plot predictions
    for pred, label in zip(predictions, labels):
        pred_sorted = pred[sort_idx]
        plt.plot(X_sorted, pred_sorted, label=label, alpha=0.8)
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Sine Wave Predictions')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/sine/predictions.png')
    plt.close()

def main():
    # Generate data
    print("Generating sine wave data...")
    X, y = generate_sine_data(n_samples=1000, noise=0.1)
    print(X.shape)
    print(y.shape)
    
    # Create models
    dt = DecisionTreeRegressor(max_depth=5)
    ft = FuzzyDecisionTreeRegressor(max_depth=9, beta=0.04, gamma=0.001)
    fbt = FuzzyBoostedTreeRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=5,
        beta=0.1,
        gamma=0.01,
        early_stopping_rounds=None  # Disable early stopping for sine wave
    )
    lgbm = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=5,
        random_state=42,
        colsample_bytree=0.8,
        subsample=0.8
    )
    
    # Train models
    print("\nTraining models...")
    dt.fit(X, y)
    ft.fit(X, y)
    fbt.fit(X, y)
    lgbm.fit(X, y)
    
    # Make predictions
    y_pred_dt = dt.predict(X)
    y_pred_ft = ft.predict(X)
    y_pred_fbt = fbt.predict(X)
    y_pred_lgbm = lgbm.predict(X)
    
    # Calculate MSE
    mse_dt = mean_squared_error(y, y_pred_dt)
    mse_ft = mean_squared_error(y, y_pred_ft)
    mse_fbt = mean_squared_error(y, y_pred_fbt)
    mse_lgbm = mean_squared_error(y, y_pred_lgbm)
    
    print("\nMean Squared Error:")
    print(f"Decision Tree: {mse_dt:.6f}")
    print(f"Fuzzy Tree: {mse_ft:.6f}")
    print(f"Fuzzy Boosted Tree: {mse_fbt:.6f}")
    print(f"LightGBM: {mse_lgbm:.6f}")
    
    # Plot results
    print("\nPlotting results...")
    plot_predictions(
        X, y,
        [y_pred_dt, y_pred_ft, y_pred_fbt, y_pred_lgbm],
        # [y_pred_dt, y_pred_ft, y_pred_lgbm],
        ['Decision Tree', 'Fuzzy Tree', 'Fuzzy Boosted Tree', 'LightGBM']
        # ['Decision Tree', 'Fuzzy Tree', 'LightGBM']
    )

if __name__ == "__main__":
    main() 