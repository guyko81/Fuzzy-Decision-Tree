import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import norm
from typing import List, Optional, Union, Tuple, Dict
from joblib import Parallel, delayed
import warnings

class FuzzyDecisionTreeRegressor:
    """A Fuzzy Decision Tree for regression tasks.
    
    This implementation uses a simple sigmoid membership function for numerical features
    and maintains crisp splits for categorical features. The fuzziness is controlled by
    a single parameter (beta) that determines the width of the fuzzy region around splits.
    """
    
    def __init__(
        self,
        max_depth: int = 5,
        beta: float = 0.1,
        categorical_features: Optional[List[int]] = None,
        random_state: Optional[int] = None,
        min_child_weight: float = 1.0,
        gamma: float = 0.0
    ):
        """Initialize the fuzzy decision tree.
        
        Args:
            max_depth: Maximum depth of the tree
            beta: Width of fuzzy regions (higher = more fuzzy)
            categorical_features: List of categorical feature indices
            random_state: Random state for reproducibility
            min_child_weight: Minimum sum of instance weights in a child node
            gamma: Minimum loss reduction required to make a further partition
        """
        self.max_depth = max_depth
        self.beta = beta
        self.categorical_features = set(categorical_features or [])
        self.random_state = random_state
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        
        # Initialize the base tree
        self.tree_ = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=max(1, int(min_child_weight)),
            random_state=random_state
        )
        
        # Store training data and membership rules
        self.X_train_ = None
        self.y_train_ = None
        self.membership_rules_ = {}  # Store membership rules for each node
        self.train_memberships_ = {}  # Store memberships for training data
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FuzzyDecisionTreeRegressor':
        """Fit the fuzzy decision tree."""
        # Store training data
        self.X_train_ = X
        self.y_train_ = y
        
        # Fit the base tree
        self.tree_.fit(X, y)
        
        # Pre-compute membership rules for all nodes
        tree = self.tree_.tree_
        for node_id in range(tree.node_count):
            if tree.children_left[node_id] != -1:  # Skip leaf nodes
                feature = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                
                # Store membership rule parameters
                self.membership_rules_[node_id] = {
                    'feature': feature,
                    'threshold': threshold,
                    'is_categorical': feature in self.categorical_features
                }
                
                # For numerical features, compute and store feature scale
                if not self.membership_rules_[node_id]['is_categorical']:
                    feature_values = X[:, feature]
                    feature_scale = np.std(feature_values) if np.std(feature_values) > 0 else 1.0
                    self.membership_rules_[node_id]['feature_scale'] = feature_scale
        
        return self
    
    def _compute_memberships(self, X: np.ndarray, node_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute left and right memberships for a node using pre-computed rules."""
        # Leaf nodes have full membership
        if node_id not in self.membership_rules_:
            return np.ones(len(X)), np.zeros(len(X))
        
        rule = self.membership_rules_[node_id]
        feature = rule['feature']
        threshold = rule['threshold']
        
        # Handle categorical features with crisp splits
        if rule['is_categorical']:
            left_membership = (X[:, feature] == threshold).astype(float)
            right_membership = 1.0 - left_membership
            return left_membership, right_membership
        
        # For numerical features, use sigmoid membership
        feature_values = X[:, feature]
        feature_scale = rule['feature_scale']
        
        # Compute normalized distance from threshold
        z = (feature_values - threshold) / (self.beta * feature_scale)
        
        # Clip to prevent numerical overflow
        z = np.clip(z, -100, 100)
        
        # Sigmoid membership function
        left_membership = 1.0 / (1.0 + np.exp(z))
        right_membership = 1.0 - left_membership
        
        # Apply min_child_weight regularization
        left_weight = np.sum(left_membership)
        right_weight = np.sum(right_membership)
        
        if left_weight < self.min_child_weight or right_weight < self.min_child_weight:
            # If either child would be too small, make the split crisp
            left_membership = (feature_values <= threshold).astype(float)
            right_membership = 1.0 - left_membership
        
        return left_membership, right_membership
    
    def _compute_optimal_value(self, memberships: np.ndarray) -> float:
        """Compute the optimal value for a node given the memberships.
        
        The optimal value minimizes the weighted squared error:
        optimal = sum(y * m) / sum(m)
        where y are the target values and m are the memberships.
        """
        # Avoid division by zero
        total_weight = np.sum(memberships)
        if total_weight < 1e-10:
            return 0.0
        
        # Compute weighted average
        return np.sum(self.y_train_ * memberships) / total_weight
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using fuzzy inference.
        
        The prediction at each leaf is weighted by the product of memberships
        along the path from root to leaf.
        """
        tree = self.tree_.tree_
        values = tree.value.reshape(-1)  # Flatten leaf values
        n_samples = X.shape[0]
        
        def traverse(node_id: int, memberships: np.ndarray) -> None:
            """Recursively traverse the tree, computing memberships."""
            # Skip empty nodes
            if node_id == -1:
                return
            
            # Get left and right memberships for current node
            left_m, right_m = self._compute_memberships(X, node_id)
            
            # Update memberships by multiplying with parent membership
            left_memberships = memberships * left_m
            right_memberships = memberships * right_m
            
            # Calculate loss reduction for this split
            if self.gamma > 0 and self.X_train_ is not None:
                # Get or compute training data memberships for this node
                if node_id not in self.train_memberships_:
                    train_left_m, train_right_m = self._compute_memberships(self.X_train_, node_id)
                    self.train_memberships_[node_id] = (train_left_m, train_right_m)
                else:
                    train_left_m, train_right_m = self.train_memberships_[node_id]
                
                # Calculate losses using optimal values
                parent_value = self._compute_optimal_value(train_left_m + train_right_m)
                left_value = self._compute_optimal_value(train_left_m)
                right_value = self._compute_optimal_value(train_right_m)
                
                # Calculate losses
                parent_loss = np.sum((self.y_train_ - parent_value)**2 * (train_left_m + train_right_m))
                left_loss = np.sum((self.y_train_ - left_value)**2 * train_left_m)
                right_loss = np.sum((self.y_train_ - right_value)**2 * train_right_m)
                loss_reduction = parent_loss - (left_loss + right_loss)
                # print(f"Parent loss: {parent_loss}, Left loss: {left_loss}, Right loss: {right_loss}, Loss reduction: {loss_reduction}")
                
                # If loss reduction is too small, make this a leaf
                if loss_reduction < self.gamma:
                    self.predictions_ += memberships * values[node_id]
                    return
            
            # If this is a leaf, add its contribution to predictions
            if tree.children_left[node_id] == -1:
                self.predictions_ += memberships * values[node_id]
                return
            
            # Traverse children
            traverse(tree.children_left[node_id], left_memberships)
            traverse(tree.children_right[node_id], right_memberships)
        
        # Initialize predictions
        self.predictions_ = np.zeros(n_samples)
        
        # Start traversal from root with full membership
        traverse(0, np.ones(n_samples))
        
        return self.predictions_