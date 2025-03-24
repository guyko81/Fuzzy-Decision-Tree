import numpy as np
from fuzzy_tree import FuzzyDecisionTreeRegressor

class FuzzyBoostedTreeRegressor:
    """Gradient Boosting with Fuzzy Decision Trees"""
    
    def __init__(self, n_estimators=100, learning_rate=0.01, max_depth=3, beta=0.1, 
                 subsample=0.8, colsample=0.8, min_child_weight=1, gamma=0.1,
                 validation_fraction=0.2, early_stopping_rounds=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.beta = beta
        self.subsample = subsample
        self.colsample = colsample
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.trees = []
        self.base_prediction = None
        self.best_iteration = None
        self.best_score = None
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Phase 1: Find optimal number of trees using validation
        if self.early_stopping_rounds is not None:
            val_size = int(n_samples * self.validation_fraction)
            indices = np.random.permutation(n_samples)
            train_indices = indices[val_size:]
            val_indices = indices[:val_size]
            
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_val = X[val_indices]
            y_val = y[val_indices]
            
            # Initialize prediction with mean
            self.base_prediction = np.mean(y_train)
            current_prediction = np.full_like(y_train, self.base_prediction)
            val_prediction = np.full_like(y_val, self.base_prediction)
            
            # Early stopping variables
            best_val_score = float('inf')
            patience_counter = 0
            optimal_trees = []
            
            # Train trees sequentially until early stopping
            for i in range(self.n_estimators):
                # Calculate residuals
                residuals = y_train - current_prediction
                
                # Subsample data for this iteration
                if self.subsample < 1.0:
                    train_subset_size = int(len(X_train) * self.subsample)
                    train_subset_indices = np.random.choice(len(X_train), train_subset_size, replace=False)
                    X_subset = X_train[train_subset_indices]
                    residuals_subset = residuals[train_subset_indices]
                else:
                    X_subset = X_train
                    residuals_subset = residuals
                
                # Feature sampling
                if self.colsample < 1.0:
                    n_features_subset = max(1, int(n_features * self.colsample))  # Ensure at least 1 feature
                    feature_indices = np.random.choice(n_features, n_features_subset, replace=False)
                    X_subset = X_subset[:, feature_indices]
                    X_train_subset = X_train[:, feature_indices]
                    X_val_subset = X_val[:, feature_indices]
                else:
                    X_train_subset = X_train
                    X_val_subset = X_val
                    feature_indices = None
                
                # Build a new fuzzy tree to predict residuals
                tree = FuzzyDecisionTreeRegressor(
                    max_depth=self.max_depth,
                    beta=self.beta,
                    min_child_weight=self.min_child_weight,
                    gamma=self.gamma
                )
                tree.fit(X_subset, residuals_subset)
                optimal_trees.append((tree, feature_indices))
                
                # Update predictions
                tree_pred_train = tree.predict(X_train_subset)
                current_prediction += self.learning_rate * tree_pred_train
                
                # Update validation predictions and check early stopping
                tree_pred_val = tree.predict(X_val_subset)
                val_prediction += self.learning_rate * tree_pred_val
                
                # Calculate validation score
                val_score = np.mean((y_val - val_prediction) ** 2)
                
                # Early stopping check
                if val_score < best_val_score:
                    best_val_score = val_score
                    self.best_iteration = i + 1
                    self.best_score = val_score
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_rounds:
                        print(f"Early stopping at iteration {i+1}")
                        break
            
            # Phase 2: Retrain on full dataset with optimal number of trees
            self.trees = []
            self.base_prediction = np.mean(y)
            current_prediction = np.full_like(y, self.base_prediction)
            
            # Train trees on full dataset
            for i in range(self.best_iteration):
                # Calculate residuals
                residuals = y - current_prediction
                
                # Subsample data for this iteration
                if self.subsample < 1.0:
                    train_subset_size = int(n_samples * self.subsample)
                    train_subset_indices = np.random.choice(n_samples, train_subset_size, replace=False)
                    X_subset = X[train_subset_indices]
                    residuals_subset = residuals[train_subset_indices]
                else:
                    X_subset = X
                    residuals_subset = residuals
                
                # Feature sampling
                if self.colsample < 1.0:
                    n_features_subset = max(1, int(n_features * self.colsample))  # Ensure at least 1 feature
                    feature_indices = np.random.choice(n_features, n_features_subset, replace=False)
                    X_subset = X_subset[:, feature_indices]
                    X_full_subset = X[:, feature_indices]
                else:
                    X_full_subset = X
                    feature_indices = None
                
                # Build a new fuzzy tree to predict residuals
                tree = FuzzyDecisionTreeRegressor(
                    max_depth=self.max_depth,
                    beta=self.beta,
                    min_child_weight=self.min_child_weight,
                    gamma=self.gamma
                )
                tree.fit(X_subset, residuals_subset)
                self.trees.append((tree, feature_indices))
                
                # Update predictions
                tree_pred = tree.predict(X_full_subset)
                current_prediction += self.learning_rate * tree_pred
        else:
            # If no early stopping, train on full dataset
            self.base_prediction = np.mean(y)
            current_prediction = np.full_like(y, self.base_prediction)
            
            # Train trees sequentially
            for i in range(self.n_estimators):
                # Calculate residuals
                residuals = y - current_prediction
                
                # Subsample data for this iteration
                if self.subsample < 1.0:
                    train_subset_size = int(n_samples * self.subsample)
                    train_subset_indices = np.random.choice(n_samples, train_subset_size, replace=False)
                    X_subset = X[train_subset_indices]
                    residuals_subset = residuals[train_subset_indices]
                else:
                    X_subset = X
                    residuals_subset = residuals
                
                # Feature sampling
                if self.colsample < 1.0:
                    n_features_subset = max(1, int(n_features * self.colsample))  # Ensure at least 1 feature
                    feature_indices = np.random.choice(n_features, n_features_subset, replace=False)
                    X_subset = X_subset[:, feature_indices]
                    X_full_subset = X[:, feature_indices]
                else:
                    X_full_subset = X
                    feature_indices = None
                
                # Build a new fuzzy tree to predict residuals
                tree = FuzzyDecisionTreeRegressor(
                    max_depth=self.max_depth,
                    beta=self.beta,
                    min_child_weight=self.min_child_weight,
                    gamma=self.gamma
                )
                tree.fit(X_subset, residuals_subset)
                self.trees.append((tree, feature_indices))
                
                # Update predictions
                tree_pred = tree.predict(X_full_subset)
                current_prediction += self.learning_rate * tree_pred
        
        return self
    
    def predict(self, X):
        # Start with base prediction
        y_pred = np.full(X.shape[0], self.base_prediction)
        
        # Add predictions from each tree
        for tree, feature_indices in self.trees:
            if feature_indices is not None:
                X_subset = X[:, feature_indices]
            else:
                X_subset = X
            y_pred += self.learning_rate * tree.predict(X_subset)
        
        return y_pred 