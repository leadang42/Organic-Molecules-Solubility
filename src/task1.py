import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from matplotlib.cm import viridis

# ======= DATA MANAGEMENT ======= # 
def create_dirs():
    """Create results directory structure"""
    os.makedirs('results/task1_noise', exist_ok=True)

def parse_experiment_id(experiment_id):
    """Converts an experiment ID to readable parameters for plot titles."""

    parts = experiment_id.split('_')
    
    n_part = parts[0] if len(parts) > 0 else ""
    sig_part = parts[1] if len(parts) > 1 else ""
    lam_part = parts[2] if len(parts) > 2 else ""
    noise_part = parts[3] if len(parts) > 3 else ""
    
    n = int(float(n_part.replace('n', '')))
    sig = float(sig_part.replace('sig', ''))
    lam = float(lam_part.replace('lam', ''))
    noise = float(noise_part.replace('noise', ''))
        
    return f"n={n}, σ={sig:.1f}, λ={lam:.3f}, noise={noise:.2f}"


# ======= FUNCTION FITTING ======= # 
def f(x):
    """Target function to fit"""
    return np.sin(x) * np.exp(x / 5)

def gaussian_kernel(x1, x2, sig):
    """Gaussian kernel"""
    return np.exp(-(x1 - x2) ** 2 / (2 * sig ** 2))

def compute_kernel_matrix(x, sig):
    """Compute kernel matrix K"""
    N = len(x)
    K = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            K[i, j] = gaussian_kernel(x[i], x[j], sig)
    return K

def fit_and_predict(x, y, xx, sig, lam):
    """Fit model and make predictions"""
    # (K + λI)  -> Regularized kernel matrix
    K = compute_kernel_matrix(x, sig)
    K_reg = K + lam * np.eye(len(x))
    
    # c = (K + λI)^(-1) y  -> Least square solution to linear matrix 
    c, _, _, _ = np.linalg.lstsq(K_reg, y, rcond=None)
    
    # y = Kc  -> Predict function on xx
    y_pred = np.zeros_like(xx)
    
    for i in range(len(xx)):
        
        # yi = Σ_j cj * K(xi, xj)
        y_pred[i] = sum(c[j] * gaussian_kernel(xx[i], x[j], sig) for j in range(len(x)))
    
    return y_pred


# ======= ANALYSIS ======= # 
def calculate_mse(y_true, y_pred):
    """Calculate mean squared error"""
    return np.mean((y_true - y_pred) ** 2)


# ======= PLOTTING ======= # 
def plot_and_save(xx, yy, x, y, y_clean, y_pred, params, mse, experiment_name):
    """Plot and save results"""
    
    # Create directory for this experiment
    save_dir = f"results/task1_noise/{experiment_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot
    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(xx, yy, color=viridis(0.3), linestyle='-', label="True function")
    plt.plot(xx, y_pred, color=viridis(0.8), linestyle='-', label="Kernel fit")
    plt.plot(x, y, color=viridis(0), marker='o', linestyle='', label="Noisy data points")
    
    # Optionally, also show the clean data points
    plt.plot(x, y_clean, color=viridis(0.5), marker='x', linestyle='', 
             alpha=0.5, label="Original data (no noise)")
    
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Kernel Regression with Noise: {parse_experiment_id(experiment_name)}\nMSE: {mse:.6f}")
    
    # Save plot
    plt.savefig(f"{save_dir}/plot.png")
    plt.close()
    
    # Convert NumPy values to standard Python types
    params = {k: float(v) if isinstance(v, np.number) else v for k, v in params.items()}
    mse = float(mse)  # Convert numpy.float64 to Python float
    
    # Save metadata
    metadata = {
        "parameters": params,
        "mse": mse,
        "num_data_points": len(x)
    }
    
    with open(f"{save_dir}/metadata.yaml", "w") as f:
        yaml.dump(metadata, f, sort_keys=False)
        
def generate_summary_plots(results):
    """Generate summary plots showing parameter relationships"""
    
    # Convert results to arrays for easy manipulation
    n_vals = np.array([r["n"] for r in results])
    sig_vals = np.array([r["sig"] for r in results])
    lam_vals = np.array([r["lam"] for r in results])
    noise_vals = np.array([r["noise"] for r in results])
    mse_vals = np.array([r["mse"] for r in results])
    
    # Set fixed values for plots
    n_fixed = 10
    sig_fixed = 1.0
    lam_fixed = 0.1
    noise_fixed = 0.1
    
    # ===== 1. Plot MSE vs. Number of Points (n) for different noise levels ===== #
    plt.figure(figsize=(7, 5), dpi=200)
    unique_noise = sorted(set(noise_vals))
    
    for i, noise in enumerate(unique_noise):
        color_idx = i / max(1, len(unique_noise) - 1)
        indices = (noise_vals == noise) & (np.abs(sig_vals - sig_fixed) < 0.001) & (np.abs(lam_vals - lam_fixed) < 0.001)
        
        # Only plot if we have enough points
        if sum(indices) >= 2:
            plt.plot(n_vals[indices], mse_vals[indices], 'o-', 
                     color=viridis(color_idx), 
                     label=f"noise={noise:.2f}")
    
    plt.xlabel("Number of Data Points (n)")
    plt.ylabel("MSE")
    plt.title(f"MSE vs. Number of Points (σ={sig_fixed}, λ={lam_fixed})")
    plt.legend()
    plt.savefig("results/task1_noise/summary_n_vs_mse.png")
    plt.close()
    
    # ===== 2. Plot MSE vs. Kernel Length Scale (sig) for different noise levels ===== #
    plt.figure(figsize=(7, 5), dpi=200)
    
    for i, noise in enumerate(unique_noise):
        color_idx = i / max(1, len(unique_noise) - 1)
        indices = (noise_vals == noise) & (np.abs(n_vals - n_fixed) < 0.1) & (np.abs(lam_vals - lam_fixed) < 0.001)
        
        # Only plot if we have enough points
        if sum(indices) >= 2:
            plt.plot(sig_vals[indices], mse_vals[indices], 'o-', 
                     color=viridis(color_idx), 
                     label=f"noise={noise:.2f}")
    
    plt.xlabel("Kernel Length Scale (σ)")
    plt.ylabel("MSE")
    plt.title(f"MSE vs. Kernel Length Scale (n={n_fixed}, λ={lam_fixed})")
    plt.xscale("log")
    plt.legend()
    plt.savefig("results/task1_noise/summary_sig_vs_mse.png")
    plt.close()
    
    # ===== 3. Plot MSE vs. Regularization (lam) for different noise levels ===== #
    plt.figure(figsize=(7, 5), dpi=200)
    
    for i, noise in enumerate(unique_noise):
        color_idx = i / max(1, len(unique_noise) - 1)
        indices = (noise_vals == noise) & (np.abs(n_vals - n_fixed) < 0.1) & (np.abs(sig_vals - sig_fixed) < 0.001)
        
        # Only plot if we have enough points
        if sum(indices) >= 2:
            plt.plot(lam_vals[indices], mse_vals[indices], 'o-', 
                     color=viridis(color_idx), 
                     label=f"noise={noise:.2f}")
    
    plt.xlabel("Regularization Parameter (λ)")
    plt.ylabel("MSE")
    plt.title(f"MSE vs. Regularization (n={n_fixed}, σ={sig_fixed})")
    plt.xscale("log")
    plt.legend()
    plt.savefig("results/task1_noise/summary_lam_vs_mse.png")
    plt.close()
    
    # ===== 4. Plot MSE vs. Noise Level for different n values ===== #
    plt.figure(figsize=(7, 5), dpi=200)
    unique_n = sorted(set(n_vals))
    
    for i, n in enumerate(unique_n):
        color_idx = i / max(1, len(unique_n) - 1)
        indices = (np.abs(n_vals - n) < 0.1) & (np.abs(sig_vals - sig_fixed) < 0.001) & (np.abs(lam_vals - lam_fixed) < 0.001)
        
        # Only plot if we have enough points
        if sum(indices) >= 2:
            plt.plot(noise_vals[indices], mse_vals[indices], 'o-', 
                     color=viridis(color_idx), 
                     label=f"n={int(n)}")
    
    plt.xlabel("Noise Level")
    plt.ylabel("MSE")
    plt.title(f"MSE vs. Noise Level (σ={sig_fixed}, λ={lam_fixed})")
    plt.legend()
    plt.savefig("results/task1_noise/summary_noise_vs_mse.png")
    plt.close()
    
    # ===== 5. Plot MSE vs. Noise Level for different sig values ===== #
    plt.figure(figsize=(7, 5), dpi=200)
    unique_sig = sorted(set(sig_vals))
    
    for i, sig in enumerate(unique_sig):
        color_idx = i / max(1, len(unique_sig) - 1)
        indices = (sig_vals == sig) & (np.abs(n_vals - n_fixed) < 0.1) & (np.abs(lam_vals - lam_fixed) < 0.001)
        
        # Only plot if we have enough points
        if sum(indices) >= 2:
            plt.plot(noise_vals[indices], mse_vals[indices], 'o-', 
                     color=viridis(color_idx), 
                     label=f"σ={sig}")
    
    plt.xlabel("Noise Level")
    plt.ylabel("MSE")
    plt.title(f"MSE vs. Noise Level with Different Kernel Scales (n={n_fixed}, λ={lam_fixed})")
    plt.legend()
    plt.savefig("results/task1_noise/summary_noise_vs_mse_by_sig.png")
    plt.close()
    
    # ===== 6. Plot MSE vs. Noise Level for different lam values ===== #
    plt.figure(figsize=(7, 5), dpi=200)
    unique_lam = sorted(set(lam_vals))
    
    for i, lam in enumerate(unique_lam):
        color_idx = i / max(1, len(unique_lam) - 1)
        indices = (lam_vals == lam) & (np.abs(n_vals - n_fixed) < 0.1) & (np.abs(sig_vals - sig_fixed) < 0.001)
        
        # Only plot if we have enough points
        if sum(indices) >= 2:
            plt.plot(noise_vals[indices], mse_vals[indices], 'o-', 
                     color=viridis(color_idx), 
                     label=f"λ={lam}")
    
    plt.xlabel("Noise Level")
    plt.ylabel("MSE")
    plt.title(f"MSE vs. Noise Level with Different Regularization (n={n_fixed}, σ={sig_fixed})")
    plt.legend()
    plt.savefig("results/task1_noise/summary_noise_vs_mse_by_lam.png")
    plt.close()
    
    # ===== 7. 3D Surface Plot: MSE as a function of σ and λ ===== #
    from mpl_toolkits.mplot3d import Axes3D
    
    plt.figure(figsize=(10, 8), dpi=200)
    ax = plt.axes(projection='3d')
    
    # Filter data for fixed n and noise
    indices = (np.abs(n_vals - n_fixed) < 0.1) & (np.abs(noise_vals - noise_fixed) < 0.001)
    
    if sum(indices) >= 4:  # Need enough points for a meaningful surface
        sig_filtered = sig_vals[indices]
        lam_filtered = lam_vals[indices]
        mse_filtered = mse_vals[indices]
        
        # Create a meshgrid for the surface
        unique_sig_filtered = sorted(set(sig_filtered))
        unique_lam_filtered = sorted(set(lam_filtered))
        
        # Only proceed if we have multiple values for both parameters
        if len(unique_sig_filtered) >= 2 and len(unique_lam_filtered) >= 2:
            X, Y = np.meshgrid(unique_sig_filtered, unique_lam_filtered)
            Z = np.zeros(X.shape)
            
            # Fill in Z values
            for i in range(len(unique_lam_filtered)):
                for j in range(len(unique_sig_filtered)):
                    lam_val = unique_lam_filtered[i]
                    sig_val = unique_sig_filtered[j]
                    
                    # Find the matching point
                    match_indices = (sig_filtered == sig_val) & (lam_filtered == lam_val)
                    if sum(match_indices) > 0:
                        Z[i, j] = mse_filtered[match_indices][0]
            
            # Plot the surface
            surf = ax.plot_surface(np.log10(X), np.log10(Y), Z, cmap='viridis', 
                                   edgecolor='none', alpha=0.8)
            
            # Add contour lines on the bottom of the plot for better readability
            ax.contour(np.log10(X), np.log10(Y), Z, zdir='z', offset=np.min(Z)*0.9, 
                       cmap='viridis', linestyles='solid')
            
            # Set labels and title
            ax.set_xlabel('log10(Kernel Length Scale σ)')
            ax.set_ylabel('log10(Regularization λ)')
            ax.set_zlabel('MSE')
            ax.set_title(f'MSE as a function of σ and λ (n={n_fixed}, noise={noise_fixed})')
            
            # Add a color bar
            plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            # Save the plot
            plt.savefig("results/task1_noise/3d_sig_lam_vs_mse.png")
    plt.close()
    
    # ===== 8. 3D Surface Plot: MSE as a function of n and noise level ===== #
    plt.figure(figsize=(10, 8), dpi=200)
    ax = plt.axes(projection='3d')
    
    # Filter data for fixed sigma and lambda
    indices = (np.abs(sig_vals - sig_fixed) < 0.001) & (np.abs(lam_vals - lam_fixed) < 0.001)
    
    if sum(indices) >= 4:  # Need enough points for a meaningful surface
        n_filtered = n_vals[indices]
        noise_filtered = noise_vals[indices]
        mse_filtered = mse_vals[indices]
        
        # Create a meshgrid for the surface
        unique_n_filtered = sorted(set(n_filtered))
        unique_noise_filtered = sorted(set(noise_filtered))
        
        # Only proceed if we have multiple values for both parameters
        if len(unique_n_filtered) >= 2 and len(unique_noise_filtered) >= 2:
            X, Y = np.meshgrid(unique_n_filtered, unique_noise_filtered)
            Z = np.zeros(X.shape)
            
            # Fill in Z values
            for i in range(len(unique_noise_filtered)):
                for j in range(len(unique_n_filtered)):
                    noise_val = unique_noise_filtered[i]
                    n_val = unique_n_filtered[j]
                    
                    # Find the matching point
                    match_indices = (n_filtered == n_val) & (noise_filtered == noise_val)
                    if sum(match_indices) > 0:
                        Z[i, j] = mse_filtered[match_indices][0]
            
            # Plot the surface
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                                   edgecolor='none', alpha=0.8)
            
            # Add contour lines on the bottom of the plot for better readability
            ax.contour(X, Y, Z, zdir='z', offset=np.min(Z)*0.9, 
                       cmap='viridis', linestyles='solid')
            
            # Set labels and title
            ax.set_xlabel('Number of Data Points (n)')
            ax.set_ylabel('Noise Level')
            ax.set_zlabel('MSE')
            ax.set_title(f'MSE as a function of n and noise level (σ={sig_fixed}, λ={lam_fixed})')
            
            # Add a color bar
            plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            # Save the plot
            plt.savefig("results/task1_noise/3d_n_noise_vs_mse.png")
    plt.close()
    
      
# ======= MAIN ======= #
def run_experiment(num_points, sig, lam, noise_level):
    """Run an experiment with specific parameters including noise level"""
    
    # Generate data
    xx = np.linspace(0, 10, 100)
    yy = f(xx)
    
    # Sample random points with fixed seed for reproducibility
    np.random.seed(42)
    x = np.random.random(num_points) * 10
    y_clean = f(x)  # Original function values without noise
    
    # Add i.i.d. noise from a normal distribution
    np.random.seed(100)  # Different seed for noise
    noise = np.random.normal(0, noise_level, num_points)
    y = y_clean + noise  # Add noise to function values
    
    # Fit and predict using the noisy data
    y_pred = fit_and_predict(x, y, xx, sig, lam)
    
    # Calculate MSE against the true function (without noise)
    mse = calculate_mse(yy, y_pred)
    
    # Create experiment name and params
    experiment_name = f"n{num_points}_sig{sig:.3f}_lam{lam:.5f}_noise{noise_level:.2f}"
    params = {
        "num_points": num_points, 
        "sig": sig, 
        "lam": lam,
        "noise_level": noise_level
    }
    
    # Plot and save
    plot_and_save(xx, yy, x, y, y_clean, y_pred, params, mse, experiment_name)
    
    return mse

def explore_parameters():
    """Explore parameters and analyze results with noise"""
    create_dirs()
    
    # Parameter ranges
    sig_values = [0.1, 1.0, 5.0, 10.0]
    lam_values = [0.001, 0.1, 1.0, 2.0]
    n_points_values = [1, 5, 10, 15, 20]
    noise_levels = [0.0, 0.2, 0.5] 
    
    results = []
    
    # Run selected experiments
    for n in n_points_values:
        for sig in sig_values:
            for lam in lam_values:
                for noise in noise_levels:
                    mse = run_experiment(n, sig, lam, noise)
                    results.append({
                        "n": float(n),
                        "sig": float(sig),
                        "lam": float(lam),
                        "noise": float(noise),
                        "mse": float(mse)
                    })
    
    best_result = min(results, key=lambda x: x["mse"])
    worst_result = max(results, key=lambda x: x["mse"]) 
    
    # Save summary
    summary = {
        "num_experiments": len(results),
        "best_parameters": best_result,
        "worst_parameters": worst_result,
    }
    
    with open("results/task1_noise/summary.yaml", "w") as f:
        yaml.dump(summary, f, sort_keys=False)
    
    # Generate summary plots
    generate_summary_plots(results)


# Main execution
if __name__ == "__main__":
    explore_parameters()
    print("Parameter exploration with noise complete. Results saved in 'results/task1_noise' directory.")