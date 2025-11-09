#!/usr/bin/env python3
"""
Hist_error_XGB.py
-----------------------------------
This script visualizes and compares the absolute prediction error distributions
of multiple trained XGBoost models based on their performance summary CSV file.

Functionality:
1. Loads the model comparison results from `model_comparison_with_errors.csv`.
2. Extracts absolute error columns (|Error_1|, |Error_2|, |Error_3|) corresponding to
   three distinct model variants.
3. Applies an upper cutoff (|Error| < 20) to exclude extreme outliers for clarity.
4. Plots logarithmic-frequency histograms to visualize the comparative error spread
   between models, highlighting model robustness and precision.

Use Case:
- Provides a visual diagnostic for regression performance comparison between
  different model architectures or training formulations.

Output:
- Displays an overlaid histogram of |Error_2| (red) and |Error_3| (green) in log-scale.
- Easily extendable to include |Error_1| by uncommenting its corresponding plot line.
"""

import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Load model comparison data
# ----------------------------
filename = "E:\\GitHub\\ML_Project\\model\\model_comparison_with_errors.csv"
data = pd.read_csv(filename)

# ----------------------------
# Extract absolute error columns
# ----------------------------
err_1 = data["|Error_1|"]
err_2 = data["|Error_2|"]
err_3 = data["|Error_3|"]

# ----------------------------
# Filter out extreme outliers for clarity
# ----------------------------
mask2 = err_2 < 20
err_2 = err_2[mask2]

mask3 = err_3 < 20
err_3 = err_3[mask3]

# ----------------------------
# Plot histograms (log-scale)
# ----------------------------
plt.figure(figsize=(10, 6))
bins = 30

# Uncomment to include Model 1 comparison
# plt.hist(err_1, bins=bins, alpha=0.6, label="Error 1", color="blue")

plt.hist(err_2, bins=bins, alpha=0.6, label="Error 2", color="red")
plt.hist(err_3, bins=bins, alpha=0.6, label="Error 3", color="green")

plt.xlabel("Absolute Error")
plt.ylabel("Frequency (log scale)")
plt.yscale("log")
plt.title("Histogram of Absolute Errors for Model Comparison")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
