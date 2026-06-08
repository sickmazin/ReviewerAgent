import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chart style configuration to make it modern and premium
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# Premium color palette (Indigo, Coral, Teal, Amber, Rose)
COLOR_TRAIN = "#4f46e5"  # Indigo
COLOR_VAL = "#f97316"    # Coral
COLOR_MAE = "#10b981"    # Emerald
COLOR_PEARSON = "#06b6d4"  # Cyan/Teal
COLOR_BG = "#f8fafc"     # Slate light background

COLORS_COMPONENTS = {
    "val_l_distill": "#6366f1",    # Violet
    "val_l_factuality": "#ec4899", # Pink
    "val_l_lexical": "#14b8a6",    # Teal
    "val_l_bound": "#f59e0b",      # Amber
    "val_l_rank": "#ef4444",       # Red
}

def plot_all(csv_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Data loading
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
        
    df = pd.read_csv(csv_path)
    batchs = df["batch"]
    
    # -------------------------------------------------------------
    # PLOT 1: Loss Curves (Train vs Validation)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLOR_BG)
    ax.set_facecolor("#ffffff")
    
    ax.plot(batchs, df["train_loss"], label="Train Loss", color=COLOR_TRAIN, linewidth=2, marker='o', markersize=4, alpha=0.85)
    ax.plot(batchs, df["val_loss"], label="Validation Loss", color=COLOR_VAL, linewidth=2, marker='s', markersize=4, alpha=0.85)
    
    # Highlight the best val_loss checkpoint
    best_idx = df["val_loss"].idxmin()
    best_batch = df.loc[best_idx, "batch"]
    best_val_loss = df.loc[best_idx, "val_loss"]
    
    ax.scatter(best_batch, best_val_loss, color="#ef4444", s=120, zorder=5, edgecolors='black', label=f"Best Checkpoint (Ep. {best_batch}: {best_val_loss:.4f})")
    ax.axvline(best_batch, color="#ef4444", linestyle="--", alpha=0.5, zorder=1)
    
    ax.set_title("Training Loss Curves", pad=15, weight="bold")
    ax.set_xlabel("batch", labelpad=10)
    ax.set_ylabel("Loss", labelpad=10)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
    ax.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "1_loss_curves.png"), bbox_inches="tight")
    plt.close(fig)
    print("Plot 1 saved: 1_loss_curves.png")

    # -------------------------------------------------------------
    # PLOT 2: Validation Metrics (MAE and Pearson R)
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6), facecolor=COLOR_BG)
    ax1.set_facecolor("#ffffff")
    
    # MAE (left y-axis)
    color = COLOR_MAE
    ax1.set_xlabel("batch", labelpad=10)
    ax1.set_ylabel("Validation MAE", color=color, labelpad=10, weight="bold")
    line1 = ax1.plot(batchs, df["val_mae"], color=color, linewidth=2, marker='o', markersize=4, label="MAE (left)")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.7)
    
    # Pearson R (right y-axis)
    ax2 = ax1.twinx()
    color = COLOR_PEARSON
    ax2.set_ylabel("Validation Pearson R", color=color, labelpad=10, weight="bold")
    line2 = ax2.plot(batchs, df["val_r"], color=color, linewidth=2, marker='^', markersize=4, label="Pearson R (right)")
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Best MAE and Pearson R
    best_mae_idx = df["val_mae"].idxmin()
    best_r_idx = df["val_r"].idxmax()
    print(f"Best Validation MAE: {df.loc[best_mae_idx, 'val_mae']:.4f} at batch {df.loc[best_mae_idx, 'batch']}")
    print(f"Best Pearson R: {df.loc[best_r_idx, 'val_r']:.4f} at batch {df.loc[best_r_idx, 'batch']}")
    
    # Merge legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
    
    plt.title("Validation Metrics (MAE & Pearson Correlation)", pad=15, weight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "2_val_metrics.png"), bbox_inches="tight")
    plt.close(fig)
    print("Plot 2 saved: 2_val_metrics.png")

    # -------------------------------------------------------------
    # PLOT 3: Validation Loss Components
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLOR_BG)
    ax.set_facecolor("#ffffff")
    
    components = ["val_l_distill", "val_l_factuality", "val_l_lexical", "val_l_bound", "val_l_rank"]
    # Filter only columns actually present
    available_components = [c for c in components if c in df.columns]
    
    for comp in available_components:
        ax.plot(batchs, df[comp], label=comp.replace("val_l_", "Loss "), 
                color=COLORS_COMPONENTS.get(comp, "#000000"), linewidth=1.8, marker='x', markersize=3, alpha=0.8)
                
    ax.set_title("Trend of Individual Validation Loss Components", pad=15, weight="bold")
    ax.set_xlabel("batch", labelpad=10)
    ax.set_ylabel("Loss Component Value", labelpad=10)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
    ax.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "3_loss_components.png"), bbox_inches="tight")
    plt.close(fig)
    print("Plot 3 saved: 3_loss_components.png")

    # -------------------------------------------------------------
    # PLOT 4: Learning Rate Behavior within batch
    # -------------------------------------------------------------
    # Explanation and simulation of CosineAnnealingWarmRestarts
    # Analysis from training.py:
    #   S (steps per batch) = 3125 (calculated as data_samples / batch_size = 20000 / 1 or 50000 / 16, etc.)
    #   For v7_frozen: S = 3125 batches (steps) per batch.
    #   scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=3125, eta_min=1e-6)
    #
    # We simulate the LR behavior on a step (batch) scale
    # to clearly show what happens within an batch.
    
    S = 3125  # steps per batch
    T_0 = 2 * S   # T_0 = 2 batchs (6250 steps)
    T_mult = 2
    eta_max = 5e-5
    eta_min = 1e-8
    
    # Generate steps for the first 5 batchs (0 to 5 * S)
    total_steps = 5 * S
    steps = np.arange(total_steps)
    lrs = np.zeros(total_steps)
    
    # Mathematical simulation of the PyTorch CosineAnnealingWarmRestarts scheduler behavior
    T_i = T_0
    T_cur = 0
    
    for step in range(total_steps):
        # Calculate LR for current step
        lrs[step] = eta_min + 0.5 * (eta_max - eta_min) * (1.0 + np.cos(T_cur / T_i * np.pi))
        
        # Advance internal scheduler counter
        T_cur += 1
        if T_cur >= T_i:
            T_cur = 0
            T_i = T_i * T_mult
            
    # Create plot with zoom
    fig, (ax_all, ax_zoom) = plt.subplots(2, 1, figsize=(10, 8), facecolor=COLOR_BG)
    ax_all.set_facecolor("#ffffff")
    ax_zoom.set_facecolor("#ffffff")
    
    # Full plot (First 5 batchs, step-by-step)
    batchs_axis = steps / S
    ax_all.plot(batchs_axis, lrs, color="#3b82f6", linewidth=1.5, label="Learning Rate (Simulated)")
    
    # Vertical lines to delimit batchs
    for ep in range(1, 6):
        ax_all.axvline(ep, color="#94a3b8", linestyle=":", alpha=0.8)
        ax_all.text(ep - 0.5, eta_max * 0.85, f"batch {ep}", ha="center", va="center", color="#475569", weight="bold", fontsize=9)
        
    ax_all.set_title("Batch-by-Batch Learning Rate Trend (First 5 batchs)", weight="bold", pad=12)
    ax_all.set_ylabel("Learning Rate", labelpad=10)
    ax_all.set_xlabel("batch (fraction)", labelpad=5)
    ax_all.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
    ax_all.grid(True, linestyle="--", alpha=0.7)
    
    # Zoomed plot around batch 2 and 3 (steps from 5500 to 7000) to see transition and restart
    zoom_start = 5500
    zoom_end = 7000
    ax_zoom.plot(steps[zoom_start:zoom_end] / S, lrs[zoom_start:zoom_end], color="#ef4444", linewidth=2, label="Transition / Restart (Zoom)")
    
    # Highlight restart at batch 2.0 (step 6250)
    ax_zoom.axvline(2.0, color="#059669", linestyle="--", alpha=0.8, label="Warm Restart (batch 2.0)")
    
    ax_zoom.set_title("Transition Detail and Warm Restart at Beginning of batch 3", weight="bold", pad=12)
    ax_zoom.set_ylabel("Learning Rate", labelpad=10)
    ax_zoom.set_xlabel("batch (fraction)", labelpad=10)
    ax_zoom.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
    ax_zoom.grid(True, linestyle="--", alpha=0.7)
    
    # Explanatory annotation
    explanation_text = (
        "NOTE:\n"
        "1. The scheduler is set with T_0 = 2 batchs (6250 steps) and T_mult = 2.\n"
        "2. The learning rate oscillates smoothly and continuously *within* each batch (batch-by-batch).\n"
        "3. The first cycle lasts exactly 2 batchs, reaching the minimum (eta_min = 1e-8) at the end of batch 2.\n"
        "4. At the beginning of batch 3, the restart to eta_max (5e-5) occurs.\n"
        "5. The second cycle lasts 2 * 2 = 4 batchs (ending at the end of batch 6)."
    )
    fig.text(0.12, 0.02, explanation_text, fontsize=9.5, color="#1e293b", 
             bbox=dict(facecolor="#f1f5f9", edgecolor="#cbd5e1", boxstyle="round,pad=0.8"))
             
    plt.subplots_adjust(bottom=0.22, hspace=0.35)
    fig.savefig(os.path.join(output_dir, "4_learning_rate_batch.png"), bbox_inches="tight")
    plt.close(fig)
    print("Plot 4 saved: 4_learning_rate_batch.png")

if __name__ == "__main__":
    # Paths relative to the script folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir,"../..","..",".models","v7_frozen","training_history.csv"))
    output_dir = os.path.abspath(os.path.join(script_dir,""))
    
    print("Starting plotting...")
    print(f"Source CSV file: {csv_path}")
    print(f"Destination folder: {output_dir}")
    
    plot_all(csv_path, output_dir)
    print("Plotting successfully completed!")
