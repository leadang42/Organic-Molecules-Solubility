#!/usr/bin/env python3
"""Kernel Ridge Regression for solubility prediction with optimized hyperparameters."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
import yaml
from datetime import datetime

# Create results directory if it doesn't exist
RESULTS_DIR = "results/task2"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===== DATA MANAGEMENT ===== #
def load_data(filepath="data/curated-solubility-dataset.csv"):
    """Load and prepare solubility dataset with normalized features."""
    print(f"Loading data from {filepath}...")
    
    sol = pd.read_csv(filepath)
    Y = np.array(sol['Solubility'])
    
    proplist = [
        'HeavyAtomCount', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds',
        'NumValenceElectrons', 'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings', 'RingCount'
    ]
    
    # Normalize features by molecular weight
    X = np.array([list(sol[prop]/sol['MolWt']) for prop in proplist])
    # Add log molecular weight as a feature
    X = np.insert(X, 0, list(np.log(sol['MolWt'])), axis=0)
    # Transpose to get shape (n_samples, n_features)
    X = X.T
    
    feature_names = ['log_MolWt'] + proplist
    return X, Y, feature_names

# ===== KERNAL FITTING ===== #

def gaussian_kernel(X1, X2, sigma):
    """Compute Gaussian kernel with different lengthscales for each dimension."""
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            diff = X1[i] - X2[j]
            K[i, j] = np.exp(-0.5 * np.sum((diff / sigma) ** 2))
    
    return K

def kernel_ridge_regression(X_train, y_train, X_test, sigma, lambda_reg):
    """Perform kernel ridge regression with Gaussian kernel."""
    n_train = X_train.shape[0]
    
    # Compute kernel matrices
    K_train = gaussian_kernel(X_train, X_train, sigma)
    K_test = gaussian_kernel(X_test, X_train, sigma)
    
    # Add regularization
    K_train_reg = K_train + lambda_reg * np.eye(n_train)
    
    # Solve the system
    alpha = np.linalg.solve(K_train_reg, y_train)
    
    # Predict
    y_pred = K_test @ alpha
    
    return y_pred

def cross_validate(X, y, sigma, lambda_reg, n_splits=5, random_state=42):
    """Perform cross-validation to evaluate model performance."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmse_scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        y_pred = kernel_ridge_regression(X_train, y_train, X_test, sigma, lambda_reg)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores.append(rmse)
    
    return np.mean(rmse_scores)

def objective_function(params, X, y, n_splits=5):
    """Objective function for hyperparameter optimization."""
    lambda_reg = params[0]
    sigma = params[1:]
    
    # Prevent negative values for hyperparameters
    if lambda_reg <= 0 or np.any(sigma <= 0):
        return 1e10
    
    return cross_validate(X, y, sigma, lambda_reg, n_splits=n_splits)

def optimize_hyperparameters(X, y, n_features, n_restarts=3):
    """Optimize regularization parameter and lengthscales."""
    print("Optimizing hyperparameters...")
    
    # Calculate standard deviation for each feature (initial sigmas)
    feature_stds = np.std(X, axis=0)
    
    best_params = None
    best_score = float('inf')
    
    for i in range(n_restarts):
        print(f"  Optimization run {i+1}/{n_restarts}")
        
        # Initial values: lambda_reg and sigmas for each feature
        initial_lambda = 10.0 ** np.random.uniform(-3, 0)  # Between 0.001 and 1.0
        initial_sigmas = feature_stds * 10.0 ** np.random.uniform(-1, 1, size=n_features)  # Around feature stds
        
        initial_params = np.concatenate(([initial_lambda], initial_sigmas))
        
        # Run optimization
        result = minimize(
            objective_function,
            initial_params,
            args=(X, y),
            method='L-BFGS-B',
            bounds=[(1e-6, 1e2)] * (n_features + 1),
            options={'maxiter': 100}
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
            
    lambda_reg = best_params[0]
    sigma = best_params[1:]
    
    print(f"  Best CV RMSE: {best_score:.4f}")
    print(f"  Best lambda: {lambda_reg:.6f}")
    print(f"  Best sigmas: {sigma}")
    
    return lambda_reg, sigma

def evaluate_model(X_train, y_train, X_test, y_test, sigma, lambda_reg):
    """Evaluate model performance on test data."""
    y_pred = kernel_ridge_regression(X_train, y_train, X_test, sigma, lambda_reg)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'rmse': float(rmse),
        'r2': float(r2)
    }
    
    return metrics, y_pred

def plot_results(y_test, y_pred, feature_names, sigma, lambda_reg, metrics):
    """Create plots for model evaluation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot predicted vs actual values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Add diagonal line
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Solubility')
    plt.ylabel('Predicted Solubility')
    plt.title(f'Kernel Ridge Regression\nRMSE={metrics["rmse"]:.4f}, R²={metrics["r2"]:.4f}')
    plt.grid(True, alpha=0.3)
    
    # Add text with hyperparameters
    textstr = f'λ = {lambda_reg:.6f}'
    plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'prediction_plot_{timestamp}.png'), dpi=300)
    
    # Plot feature importance (based on 1/sigma² values)
    importance = 1 / (sigma ** 2)
    importance = importance / np.sum(importance)  # Normalize
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, importance)
    plt.xlabel('Features')
    plt.ylabel('Relative Importance (1/σ²)')
    plt.title('Feature Importance Based on Optimized Lengthscales')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'feature_importance_{timestamp}.png'), dpi=300)
    
    plt.close('all')

def save_metadata(sigma, lambda_reg, feature_names, metrics, train_size, test_size):
    """Save metadata of the experiment to a YAML file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    metadata = {
        'timestamp': timestamp,
        'model': 'Kernel Ridge Regression with Gaussian Kernel',
        'hyperparameters': {
            'lambda': float(lambda_reg),
            'sigma': {feature: float(s) for feature, s in zip(feature_names, sigma)}
        },
        'metrics': metrics,
        'dataset': {
            'training_samples': int(train_size),
            'test_samples': int(test_size)
        }
    }
    
    filepath = os.path.join(RESULTS_DIR, f'metadata_{timestamp}.yaml')
    with open(filepath, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    print(f"Metadata saved to {filepath}")

def main():
    """Main execution function."""
    # Load and prepare data
    X, y, feature_names = load_data()
    n_samples, n_features = X.shape
    
    print(f"Dataset: {n_samples} samples, {n_features} features")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Optimize hyperparameters
    lambda_reg, sigma = optimize_hyperparameters(X_train, y_train, n_features)
    
    # Evaluate on test set
    metrics, y_pred = evaluate_model(X_train, y_train, X_test, y_test, sigma, lambda_reg)
    
    print("\nTest set performance:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    
    # Create plots
    plot_results(y_test, y_pred, feature_names, sigma, lambda_reg, metrics)
    
    # Save metadata
    save_metadata(sigma, lambda_reg, feature_names, metrics, X_train.shape[0], X_test.shape[0])

if __name__ == "__main__":
    main()