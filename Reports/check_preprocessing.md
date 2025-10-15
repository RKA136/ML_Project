# Verification of Calorimeter Event Feature Extraction

## 1. Introduction

To ensure the correctness of GPU-accelerated feature extraction, a verification is performed by comparing manually computed features for a single event with the preprocessed tensor features. This step validates the integrity and accuracy of the data pipeline before model training.

---

## 2. Loading Preprocessed Tensors

The preprocessed features and labels are stored in PyTorch format. They are loaded safely on CPU to prevent GPU memory overflow. The data structure consists of:

- **X_tensor**: Feature matrix with shape `(n_events, n_features)`
- **y_tensor**: True energy labels with shape `(n_events, 1)`

The keys and shapes of the loaded dictionary are printed to confirm successful loading.

---

## 3. Manual Feature Computation for a Single Event

For the first event, features are computed manually from raw HDF5 hit data. The computation includes:

1. **Energy sums and maxima**:
   - Total energy deposited in the event.
   - Maximum energy among individual hits.

2. **Energy-weighted center-of-gravity (COG)**:
   - Compute the x, y, z coordinates of the energy-weighted center.

3. **Weighted standard deviations**:
   - Radial spread around the COG.
   - Longitudinal spread along the z-axis.

4. **Radius containing 90% of energy (r90)**:
   - Hits are sorted by radial distance from the COG.
   - The radius at which cumulative energy reaches 90% of total energy is determined.

5. **Layer-wise energy fractions**:
   - Assign each hit to a calorimeter layer based on its z-coordinate.
   - Compute fraction of total energy deposited in each layer.

The manually computed features are concatenated to form a complete feature vector for the event.

---

## 4. Comparison with Preprocessed Tensor Features

The manually computed feature vector is compared element-wise with the corresponding row from **X_tensor**:

- Absolute differences between manual and preprocessed features are computed.
- Features with differences below a tolerance (1e-3) are considered matching.
- Features exceeding the tolerance are flagged as mismatches.

**Observations:**

- Total energy, maximum energy, weighted standard deviations, r90, and per-layer energy fractions match closely.
- Differences are consistently below 1e-3, confirming correctness.
- This verification provides confidence that the GPU-accelerated feature pipeline reproduces accurate physics-based features.

---