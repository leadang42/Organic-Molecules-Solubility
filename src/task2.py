#!/usr/bin/env python3
"""Kernel Ridge Regression for solubility prediction with optimized hyperparameters."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
from scipy.stats import pearsonr
import yaml
from datetime import datetime
import time
import multiprocessing
from functools import partial
from joblib import Parallel, delayed

# Create results directory if it doesn't exist
RESULTS_DIR = "results/task2"
os.makedirs(RESULTS_DIR, exist_ok=True)

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
    return X, Y, feature_names, sol

def gaussian_kernel(X1, X2, sigma):
    """Compute Gaussian kernel with different lengthscales for each dimension (fully vectorized)."""
    # Normalize features by their respective sigma values
    X1_normalized = X1 / sigma
    X2_normalized = X2 / sigma
    
    # Fully vectorized implementation
    # This approach uses broadcasting to compute all distances at once
    # It's much faster but uses more memory
    n1, n2 = X1_normalized.shape[0], X2_normalized.shape[0]
    
    # Compute squared differences for each pair of points and each feature
    # Reshape to broadcast properly: (n1, 1, d) - (1, n2, d) -> (n1, n2, d)
    sq_diff = (X1_normalized[:, np.newaxis, :] - X2_normalized[np.newaxis, :, :]) ** 2
    
    # Sum over features (axis=2) and apply exp
    K = np.exp(-0.5 * np.sum(sq_diff, axis=2))
    
    return K

def kernel_ridge_regression(X_train, y_train, X_test, sigma, lambda_reg):
    """Perform kernel ridge regression with Gaussian kernel."""
    n_train = X_train.shape[0]
    
    # For very large datasets, process kernel matrices in chunks to save memory
    if n_train > 5000:
        # Process in chunks for large datasets
        chunk_size = 2000
        K_train = np.zeros((n_train, n_train))
        
        # Compute kernel matrix in chunks
        for i in range(0, n_train, chunk_size):
            end_i = min(i + chunk_size, n_train)
            X_chunk = X_train[i:end_i]
            K_train[i:end_i] = gaussian_kernel(X_chunk, X_train, sigma)
    else:
        # Compute kernel matrices
        K_train = gaussian_kernel(X_train, X_train, sigma)
    
    # Add regularization
    K_train_reg = K_train + lambda_reg * np.eye(n_train)
    
    # Solve the system (use a more stable method for large matrices)
    if n_train > 5000:
        # Use Cholesky decomposition for stability with large matrices
        L = np.linalg.cholesky(K_train_reg)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    else:
        alpha = np.linalg.solve(K_train_reg, y_train)
    
    # Compute test kernel matrix
    K_test = gaussian_kernel(X_test, X_train, sigma)
    
    # Predict
    y_pred = K_test @ alpha
    
    return y_pred

def cross_validate(X, y, sigma, lambda_reg, n_splits=5, random_state=42):
    """Perform cross-validation to evaluate model performance."""
    # For very large datasets, use a subset of folds to speed up computation
    n_samples = X.shape[0]
    if n_samples > 5000 and n_splits > 3:
        n_splits = 3  # Use fewer folds for very large datasets
    
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

def callback_factory(X, y):
    """Create a callback function with persistent iteration counter."""
    data = {'iter': 0, 'best_score': float('inf'), 'start_time': time.time()}
    
    def callback(params):
        data['iter'] += 1
        score = objective_function(params, X, y)
        elapsed = time.time() - data['start_time']
        
        if score < data['best_score']:
            data['best_score'] = score
            print(f"    Iteration {data['iter']:3d}: New best RMSE = {score:.4f} (elapsed: {elapsed:.1f}s)")
        if data['iter'] % 5 == 0:
            print(f"    Iteration {data['iter']:3d}: Current RMSE = {score:.4f} (elapsed: {elapsed:.1f}s)")
    
    return callback, data

def optimize_hyperparameters(X, y, n_features, n_restarts=3, n_folds=5, max_iter=100):
    """Optimize regularization parameter and lengthscales."""
    print(f"Optimizing hyperparameters with {n_folds}-fold CV, {n_restarts} restarts, {max_iter} max iterations...")
    
    # Calculate standard deviation for each feature (initial sigmas)
    feature_stds = np.std(X, axis=0)
    
    best_params = None
    best_score = float('inf')
    optimization_history = []
    
    # Try starting with different hyperparameter values for each run
    # Define specific starting points for faster convergence
    initial_lambdas = [0.05, 0.1, 0.01]  # Based on previous good results
    
    for i in range(n_restarts):
        print(f"  Optimization run {i+1}/{n_restarts}")
        start_time = time.time()
        
        # Use predefined lambda values for early runs, then random for later runs
        if i < len(initial_lambdas):
            initial_lambda = initial_lambdas[i]
        else:
            initial_lambda = 10.0 ** np.random.uniform(-3, 0)
            
        # For sigmas, initialize near feature standard deviations with some variation
        initial_sigmas = feature_stds * (0.5 + np.random.rand(n_features))
        
        initial_params = np.concatenate(([initial_lambda], initial_sigmas))
        
        # Create callback for this optimization run
        callback_fn, callback_data = callback_factory(X, y)
        
        # Run optimization with parallel processing where possible
        result = minimize(
            objective_function,
            initial_params,
            args=(X, y, n_folds),
            method='L-BFGS-B',
            bounds=[(1e-6, 1e2)] * (n_features + 1),
            options={'maxiter': max_iter, 'ftol': 1e-4},  # Less strict tolerance for faster convergence
            callback=callback_fn
        )
        
        run_time = time.time() - start_time
        print(f"  Run {i+1} completed in {run_time:.1f} seconds. Final RMSE: {result.fun:.4f}")
        
        # Store optimization history
        optimization_history.append({
            'run': i+1,
            'final_rmse': float(result.fun),
            'lambda': float(result.x[0]),
            'runtime': float(run_time)
        })
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
    
    lambda_reg = best_params[0]
    sigma = best_params[1:]
    
    print(f"  Best CV RMSE: {best_score:.4f}")
    print(f"  Best lambda: {lambda_reg:.6f}")
    print(f"  Best sigmas: {sigma}")
    
    return lambda_reg, sigma, optimization_history

def evaluate_model(X_train, y_train, X_test, y_test, sigma, lambda_reg):
    """Evaluate model performance on test data."""
    y_pred = kernel_ridge_regression(X_train, y_train, X_test, sigma, lambda_reg)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    
    metrics = {
        'rmse': float(rmse),
        'r2': float(r2),
        'pearson_correlation': float(pearson_corr)
    }
    
    errors = y_test - y_pred
    metrics['mean_absolute_error'] = float(np.mean(np.abs(errors)))
    metrics['max_error'] = float(np.max(np.abs(errors)))
    
    return metrics, y_pred

def plot_predicted_vs_actual(y_test, y_pred, lambda_reg, metrics, timestamp):
    """Create a scatter plot of predicted vs actual values with improved styling."""
    plt.figure(figsize=(10, 8))
    
    # Create a colormap based on data density
    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=0, vmax=0.8)  # Using viridis with max 0.8 as requested
    
    # Create scatter plot with colored points
    plt.scatter(y_test, y_pred, alpha=0.7, c=np.abs(y_test - y_pred), cmap=cmap, norm=norm)
    
    # Add color bar
    cbar = plt.colorbar()
    cbar.set_label('Absolute Prediction Error')
    
    # Add diagonal line
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')
    
    # Add labels and title
    plt.xlabel('Actual Solubility (log mol/L)', fontsize=12)
    plt.ylabel('Predicted Solubility (log mol/L)', fontsize=12)
    plt.title('Kernel Ridge Regression: Predicted vs Actual Solubility', fontsize=14)
    
    # Add statistics as text box
    textstr = "\n".join([
        f'RMSE = {metrics["rmse"]:.4f}',
        f'R² = {metrics["r2"]:.4f}',
        f'Pearson r = {metrics["pearson_correlation"]:.4f}',
        f'λ = {lambda_reg:.6f}'
    ])
    
    plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                 fontsize=11, verticalalignment='top')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(RESULTS_DIR, f'prediction_plot_{timestamp}.png'), dpi=300)

def plot_feature_importance(feature_names, sigma, timestamp):
    """Create a bar plot of feature importance based on lengthscales."""
    # Calculate importance based on 1/sigma²
    importance = 1 / (sigma ** 2)
    importance = importance / np.sum(importance)  # Normalize
    
    plt.figure(figsize=(12, 7))
    
    # Sort features by importance
    sorted_indices = np.argsort(importance)
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importance = importance[sorted_indices]
    
    # Create horizontal bar chart with viridis colormap
    bars = plt.barh(sorted_features, sorted_importance, color=plt.cm.viridis(np.linspace(0.1, 0.8, len(sorted_features))))
    
    # Add values to the end of each bar
    for i, v in enumerate(sorted_importance):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.xlabel('Relative Importance (1/σ²)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Feature Importance Based on Optimized Lengthscales', fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    plt.savefig(os.path.join(RESULTS_DIR, f'feature_importance_{timestamp}.png'), dpi=300)

def plot_error_distribution(y_test, y_pred, timestamp):
    """Create a histogram of prediction errors."""
    errors = y_test - y_pred
    
    plt.figure(figsize=(10, 6))
    
    # Create histogram with viridis colormap
    n, bins, patches = plt.hist(errors, bins=30, alpha=0.7, color=plt.cm.viridis(0.6))
    
    # Add a normal distribution fit
    mu, sigma = np.mean(errors), np.std(errors)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = len(errors) * (bins[1] - bins[0]) * 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))
    plt.plot(x, y, 'r--', linewidth=2, label=f'Normal: μ={mu:.3f}, σ={sigma:.3f}')
    
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Zero Error')
    plt.axvline(x=mu, color='red', linestyle='-', alpha=0.5, label=f'Mean Error: {mu:.3f}')
    
    plt.xlabel('Prediction Error (Actual - Predicted)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Errors', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(RESULTS_DIR, f'error_distribution_{timestamp}.png'), dpi=300)

def plot_residuals(y_test, y_pred, timestamp):
    """Create a residual plot to check for patterns in errors."""
    residuals = y_test - y_pred
    
    plt.figure(figsize=(10, 6))
    
    # Use viridis colormap based on absolute residual values
    sc = plt.scatter(y_pred, residuals, alpha=0.7, 
                    c=np.abs(residuals), cmap='viridis', norm=Normalize(vmin=0, vmax=3))
    
    plt.axhline(y=0, color='r', linestyle='--')
    plt.colorbar(sc, label='Absolute Residual')
    
    # Add a LOWESS trend line if scipy.stats is available
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        trend = lowess(residuals, y_pred, frac=0.3)
        plt.plot(trend[:, 0], trend[:, 1], 'r-', linewidth=2, label='LOWESS Trend')
        plt.legend()
    except ImportError:
        pass
    
    plt.xlabel('Predicted Solubility', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.title('Residual Plot', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(RESULTS_DIR, f'residuals_plot_{timestamp}.png'), dpi=300)

def plot_optimization_history(history, timestamp):
    """Create a plot showing optimization progress across different runs."""
    plt.figure(figsize=(10, 6))
    
    # Extract data
    runs = [h['run'] for h in history]
    final_rmse = [h['final_rmse'] for h in history]
    runtimes = [h['runtime'] for h in history]
    lambdas = [h['lambda'] for h in history]
    
    # Create two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot RMSE on the primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Optimization Run', fontsize=12)
    ax1.set_ylabel('Final RMSE', fontsize=12, color=color)
    ax1.bar(runs, final_rmse, color=plt.cm.viridis(0.3), alpha=0.7, label='Final RMSE')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Add RMSE values as text on top of bars
    for i, v in enumerate(final_rmse):
        ax1.text(i+1, v + 0.02, f'{v:.4f}', ha='center')
    
    # Create a secondary y-axis for runtime
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Runtime (seconds)', fontsize=12, color=color)
    ax2.plot(runs, runtimes, 'o-', color=color, label='Runtime')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add lambda values as annotations
    for i, v in enumerate(lambdas):
        ax1.text(i+1, final_rmse[i] - 0.1, f'λ={v:.4f}', ha='center', rotation=90, fontsize=8)
    
    plt.title('Optimization Performance by Run', fontsize=14)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'optimization_history_{timestamp}.png'), dpi=300)
    plt.close(fig)

def save_metadata(sigma, lambda_reg, feature_names, metrics, 
                  train_size, test_size, optimization_history, timestamp):
    """Save metadata of the experiment to a YAML file."""
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
        },
        'optimization_history': optimization_history
    }
    
    filepath = os.path.join(RESULTS_DIR, f'metadata_{timestamp}.yaml')
    with open(filepath, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    print(f"Metadata saved to {filepath}")

def main():
    """Main execution function."""
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    print(f"Starting execution at {start_time.strftime('%H:%M:%S')}")
    
    # Load and prepare data
    X, y, feature_names, raw_data = load_data()
    n_samples, n_features = X.shape
    
    print(f"Dataset: {n_samples} samples, {n_features} features")
    
    # For parameter optimization, use a smaller subset but keep full dataset for final model
    opt_sample_size = 2000  # Use a smaller subset for hyperparameter optimization
    print(f"Using {opt_sample_size} samples for hyperparameter optimization...")
    
    np.random.seed(42)  # For reproducibility
    opt_indices = np.random.choice(n_samples, opt_sample_size, replace=False)
    X_opt = X[opt_indices]
    y_opt = y[opt_indices]
    
    # Split optimization data for cross-validation
    X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
        X_opt, y_opt, test_size=0.2, random_state=42
    )
    
    # Split full data into train and test sets for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Optimization training set: {X_opt_train.shape[0]} samples")
    print(f"Final training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Optimize hyperparameters with optimization subset
    lambda_reg, sigma, optimization_history = optimize_hyperparameters(
        X_opt_train, y_opt_train, n_features, 
        n_restarts=3,     # Slightly reduced but still more than original
        n_folds=5,        
        max_iter=80       # Slightly reduced but still much higher than original
    )
    
    print("Re-training final model on full training dataset with optimized parameters...")
    
    # Evaluate on test set
    metrics, y_pred = evaluate_model(X_train, y_train, X_test, y_test, sigma, lambda_reg)
    
    print("\nTest set performance:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Pearson correlation: {metrics['pearson_correlation']:.4f}")
    print(f"  Mean absolute error: {metrics['mean_absolute_error']:.4f}")
    print(f"  Maximum error: {metrics['max_error']:.4f}")
    
    # Create plots
    plot_predicted_vs_actual(y_test, y_pred, lambda_reg, metrics, timestamp)
    plot_feature_importance(feature_names, sigma, timestamp)
    plot_error_distribution(y_test, y_pred, timestamp)
    plot_residuals(y_test, y_pred, timestamp)
    plot_optimization_history(optimization_history, timestamp)
    
    # Save metadata
    save_metadata(sigma, lambda_reg, feature_names, metrics, 
                 X_train.shape[0], X_test.shape[0], optimization_history, timestamp)
    
    # Report total execution time
    end_time = datetime.now()
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time}")

if __name__ == "__main__":
    main()