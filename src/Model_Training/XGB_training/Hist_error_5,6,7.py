#!/usr/bin/env python3
"""
Hist_error_XGB_v7.py
-----------------------------------
Visualizes and compares the absolute prediction error distributions
for the XGBoost models v5, v6, and v7.

Functionality:
--------------
1. Loads the model comparison results from `model/model_comparison_v5_v7.csv`.
2. Extracts the absolute error columns (|Error_v5|, |Error_v6|, |Error_v7|).
3. Applies an upper cutoff (|Error| < 20) to remove extreme outliers for clearer visualization.
4. Plots overlaid histograms (logarithmic y-scale) to show comparative precision
   and robustness between model versions.

Use Case:
---------
Provides a direct, visual diagnostic of model performance improvements
from direct regression (v5) to ratio-based (v6) to log-ratio-based (v7) targets.

Output:
-------
- Displays overlaid histograms of |Error_v5| (blue), |Error_v6| (red), and |Error_v7| (green)
  in logarithmic frequency scale.
- Can be extended to save figures to disk.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ================================================================
# Configuration
# ================================================================
MODEL_DIR = "model"
CSV_FILENAME = os.path.join(MODEL_DIR, "model_comparison_v5_v7.csv")
SAVE_FIG = True
OUT_FIG_PATH = os.path.join(MODEL_DIR, "hist_error_v5_v7.png")
ERROR_CUTOFF = 20
BINS = 40

# ================================================================
# Load data
# ================================================================
if not os.path.exists(CSV_FILENAME):
    raise FileNotFoundError(f"Comparison CSV not found: {CSV_FILENAME}")

data = pd.read_csv(CSV_FILENAME)

# ================================================================
# Extract and filter absolute errors
# ================================================================
abs_err_v5 = data["|Error_v5|"]
abs_err_v6 = data["|Error_v6|"]
abs_err_v7 = data["|Error_v7|"]

# Filter extreme outliers
abs_err_v5 = abs_err_v5[abs_err_v5 < ERROR_CUTOFF]
abs_err_v6 = abs_err_v6[abs_err_v6 < ERROR_CUTOFF]
abs_err_v7 = abs_err_v7[abs_err_v7 < ERROR_CUTOFF]

# ================================================================
# Plot histograms
# ================================================================
plt.figure(figsize=(10, 6))

plt.hist(abs_err_v5, bins=BINS, alpha=0.5, label="|Error_v5| (Direct Energy)", color="blue")
plt.hist(abs_err_v6, bins=BINS, alpha=0.5, label="|Error_v6| (Ratio-based)", color="red")
plt.hist(abs_err_v7, bins=BINS, alpha=0.5, label="|Error_v7| (Log-Ratio-based)", color="green")

plt.yscale("log")
plt.xlabel("Absolute Error (|E_pred - E_true|)")
plt.ylabel("Frequency (log scale)")
plt.title("Histogram of Absolute Errors (v5 vs v6 vs v7)")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.tight_layout()

# Save or show
if SAVE_FIG:
    plt.savefig(OUT_FIG_PATH, dpi=200)
    print(f"Saved histogram figure to: {OUT_FIG_PATH}")

plt.show()
