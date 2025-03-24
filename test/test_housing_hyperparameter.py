import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import sys
import os

# Add parent directory to path to import fuzzy_tree
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fuzzy_tree import FuzzyDecisionTreeRegressor

def plot_feature_comparison(X, y, feature_idx, feature_name, models, predictions, scaler):
    """Plot predictions vs a specific feature"""
    plt.figure(figsize=(10, 6))
    
    # Sort by feature values
    sort_idx = np.argsort(X[:, feature_idx])
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    # Plot actual values
    plt.scatter(X_sorted[:, feature_idx], y_sorted, color='gray', alpha=0.5, label='Actual')
    
    # Plot predictions
    colors = ['b-', 'g-', 'r-', 'y-']
    for pred, label, color in zip(predictions, models, colors):
        pred_sorted = pred[sort_idx]
        plt.plot(X_sorted[:, feature_idx], pred_sorted, color, label=label, alpha=0.8)
    
    plt.xlabel(feature_name)
    plt.ylabel('House Value')
    plt.title(f'Predictions vs {feature_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/housing/feature_comparison_{feature_idx}.png')
    plt.close()

def main():
    print("Loading California housing dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create grid of parameters
    depths = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    betas = [0.04, 0.06, 0.08, 0.1, 0.12, 0.14]
    
    # Create models
    models = {}
    
    # Add Decision Trees with different depths
    for depth in depths:
        name = f"Decision Tree (depth={depth})"
        models[name] = DecisionTreeRegressor(
            max_depth=depth,
            min_samples_split=5,
            min_samples_leaf=1
        )
    
    # Add Fuzzy Trees with different parameters
    for depth in depths:
        for beta in betas:
            name = f"Fuzzy Tree (β={beta:.2f}, depth={depth})"
            models[name] = FuzzyDecisionTreeRegressor(
                max_depth=depth,
                beta=beta,
                min_child_weight=1,
                gamma=0.0
            )
    
    # Train models and collect predictions
    predictions = {}
    times = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        times[name] = time.time() - start_time
        
        # Make predictions
        predictions[name] = model.predict(X_test_scaled)
    
    # Calculate metrics
    print("\nResults:")
    for name in models:
        mse = mean_squared_error(y_test, predictions[name])
        r2 = r2_score(y_test, predictions[name])
        print(f"\n{name}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  R2 Score: {r2:.4f}")
        print(f"  Training Time: {times[name]:.2f} seconds")
    
    # Plot feature comparisons for best models
    print("\nPlotting feature comparisons...")
    important_features = [0, 4, 5]  # MedInc, Population, AveOccup
    
    # Find best Decision Tree and Fuzzy Tree
    best_dt = min((name for name in models if "Decision Tree" in name), 
                  key=lambda x: mean_squared_error(y_test, predictions[x]))
    best_ft = min((name for name in models if "Fuzzy Tree" in name), 
                  key=lambda x: mean_squared_error(y_test, predictions[x]))
    
    print(f"\nBest Decision Tree: {best_dt}")
    print(f"Best Fuzzy Tree: {best_ft}")
    
    for i in important_features:
        print(f"Plotting comparison for {feature_names[i]}...")
        plot_feature_comparison(
            X_test_scaled, y_test, i, feature_names[i],
            [best_dt, best_ft],
            [predictions[best_dt], predictions[best_ft]],
            scaler
        )

if __name__ == "__main__":
    main() 