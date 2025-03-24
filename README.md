# Fuzzy Decision Trees and Random Forests

A scikit-learn compatible implementation of fuzzy decision trees and random forests that support both numerical and categorical features. This implementation provides a smooth, fuzzy alternative to traditional decision trees by using sigmoid-based membership functions for numerical features and exact matching for categorical features.

## Features

- Scikit-learn compatible API
- Support for both numerical and categorical features
- Sigmoid-based fuzzy membership for numerical features
- Exact matching for categorical features
- Both single tree and random forest implementations
- Parallel processing support for random forests
- Feature importance computation

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Single Fuzzy Decision Tree

```python
from fuzzy_tree import FuzzyDecisionTreeRegressor

# Create and fit a fuzzy decision tree
tree = FuzzyDecisionTreeRegressor(
    max_depth=5,                    # Maximum tree depth
    categorical_features=[3, 4],    # Indices of categorical features
    sigma_factor=0.01,             # Controls width of fuzzy regions
    steepness=5.0,                 # Controls sharpness of transitions
    random_state=42
)

# Fit the model
tree.fit(X_train, y_train)

# Make predictions
y_pred = tree.predict(X_test)
```

### Fuzzy Random Forest

```python
from fuzzy_tree import FuzzyRandomForestRegressor

# Create and fit a fuzzy random forest
forest = FuzzyRandomForestRegressor(
    n_estimators=100,              # Number of trees
    max_depth=5,                   # Maximum tree depth
    categorical_features=[3, 4],    # Indices of categorical features
    sigma_factor=0.01,             # Controls width of fuzzy regions
    steepness=5.0,                 # Controls sharpness of transitions
    max_features='sqrt',           # Number of features to consider for splits
    n_jobs=-1,                     # Use all available cores
    random_state=42
)

# Fit the model
forest.fit(X_train, y_train)

# Make predictions
y_pred = forest.predict(X_test)

# Get feature importances
importances = forest.feature_importances_
```

## Parameters

### Common Parameters

- `max_depth`: Maximum depth of the tree(s)
- `min_samples_split`: Minimum samples required to split a node
- `min_samples_leaf`: Minimum samples required at a leaf node
- `categorical_features`: List of indices of categorical features
- `sigma_factor`: Controls the width of fuzzy regions around split points
- `steepness`: Controls how sharp the transitions are between regions
- `random_state`: Random state for reproducibility

### Additional Random Forest Parameters

- `n_estimators`: Number of trees in the forest
- `max_features`: Number of features to consider for best split
- `bootstrap`: Whether to use bootstrap sampling
- `n_jobs`: Number of parallel jobs

## How It Works

1. For numerical features:
   - Uses a sigmoid function to create smooth transitions around split points
   - `sigma_factor` controls the width of the transition region
   - `steepness` controls how sharp the transition is

2. For categorical features:
   - Uses exact matching (no fuzziness)
   - Automatically handles string categories by mapping them to numeric values

3. Prediction process:
   - Computes fuzzy memberships for each node
   - Combines memberships along paths to leaves
   - Returns weighted average of leaf values based on memberships

4. Random Forest:
   - Creates multiple fuzzy trees using bootstrap sampling
   - Averages predictions across all trees
   - Provides feature importance based on usage in splits

## Example

See `example.py` for a complete example using synthetic data with both numerical and categorical features. 