# GPU-Accelerated Event Feature Computation Report

## Overview

This Python code processes calorimeter event data from HDF5 files and computes per-event features using GPU acceleration (CuPy). These features are intended for machine learning applications, capturing both total energy, hit distributions, and per-layer energy deposition patterns. Features are returned as PyTorch tensors.

---

## Features Computed

### 1. Total Event Energy ($E_{sum}$)

**Importance:**  
Represents the total energy deposited by an event in the calorimeter. Fundamental for regression tasks predicting the total particle energy.  

**Mathematical Formulation:**  
$E_{sum}^i = Σ_{j=1}^{N_i} E_{ij}  $
Where $N_i$ is the number of hits in event i, and $E_{ij}$ is the energy of hit j.

**Code Snippet:**  
```python
E_sum = cp.bincount(event_ids, weights=E, minlength=batch_end - batch_start)  
E_sum = cp.maximum(E_sum, 1e-8)
```

---

### 2. Maximum Hit Energy ($E_{max}$)

**Importance:**  
Captures the highest single-hit energy. Helps detect sharp energy spikes, which may indicate specific particle interactions.  

**Mathematical Formulation:**  
$E_{max}^i = max_{j=1..N_i} E_{ij}$

**Code Snippet:**  
```python
E_max = cp.zeros(batch_end - batch_start, dtype=cp.float32)  
cp.maximum.at(E_max, event_ids, E)
```

---

### 3. Energy-weighted Center-of-Gravity Standard Deviation in Radial Direction ($r_{std}$)

**Importance:**  
Measures how spread out the hits are around the energy-weighted center in the transverse plane. Sensitive to shower shape.  

**Mathematical Formulation:**  
1. Compute COG:  
$x_{cog} = Σ_j E_j x_j / E_{sum} \quad y_{cog} = Σ_j E_j y_j / E_{sum}  $
2. Radial distance of hit j:  
$r_j = \sqrt{(x_j - x_{cog})^2 + (y_j - y_{cog})^2}$  
3. Weighted standard deviation:  
$r_{std} = \sqrt{\frac{Σ_j E_j r_j^2}{E_{sum}} - (\frac{Σ_j E_j r_j}{E_{sum}})^2}$

**Code Snippet:**  
```python
def weighted_std(vals):  
    mean_sq = cp.bincount(event_ids, weights=vals**2 * E, minlength=batch_end - batch_start)/E_sum  
    mean_val = cp.bincount(event_ids, weights=vals * E, minlength=batch_end - batch_start)/E_sum  
    return cp.sqrt(cp.maximum(mean_sq - mean_val**2, 0)) 
 
r_std = weighted_std(r)  
```
---

### 4. Energy-weighted Center-of-Gravity Standard Deviation in Z ($z_{std}$)

**Importance:**  
Measures the longitudinal spread of the hits around the energy-weighted COG. Useful to understand how energy propagates along the calorimeter depth.  

**Mathematical Formulation:**  
$z_{std} = \sqrt{(\frac{Σ_j E_j (z_j - z_{cog})^2}{E_{sum}} - (\frac{Σ_j E_j (z_j - z_{cog})}{E_{sum}})^2)}$

**Code Snippet:**  
```python
z_std = weighted_std(z_shift)  
```
---

### 5. 90% Energy Containment Radius (r90)

**Importance:**  
Describes the compactness of the event in the transverse plane; radius containing 90% of the event energy. Helps distinguish narrow vs. wide showers.  

**Mathematical Formulation:**  
1. Sort radial distances $r_j$ in ascending order for each event.  
2. Compute cumulative energy: $C_k = Σ_{j=1}^k E_j$  
3. Find r90 as the smallest radius containing 90% of total event energy:  
$r90 = r_k$ such that $C_k ≥ 0.9 E_{sum}$

**Code Snippet:**  
```python
r90 = cp.zeros(batch_end - batch_start, dtype=cp.float32)  
for i in range(batch_end - batch_start):  
    mask = event_ids == i  
    if cp.any(mask):  
        r_ev = r[mask]  
        E_ev = E[mask]  
        order = cp.argsort(r_ev)  
        r_sorted = r_ev[order]  
        E_sorted = E_ev[order]  
        cumE = cp.cumsum(E_sorted)  
        idx90 = cp.searchsorted(cumE, 0.9 * cumE[-1])  
        r90[i] = r_sorted[min(idx90, r_sorted.size-1)]  
```
---

### 6. Energy Fraction per Layer (E_layer_frac[i])

**Importance:**  
Represents the fraction of total energy deposited in each calorimeter layer. Helps the model learn depth-dependent shower behavior.  

**Mathematical Formulation:**  
$E_{layer-frac}^i = \frac{Σ_{\text{(j in layer i)}} E_j}{E_{sum}}$

**Code Snippet:**  
```python
linear_idx = event_ids * n_layers + layer_idx  
E_layer_sum = cp.bincount(linear_idx, weights=E, minlength=(batch_end - batch_start) * n_layers)  
E_layer_sum = E_layer_sum.reshape(batch_end - batch_start, n_layers)  
E_layer_frac = E_layer_sum / E_sum[:, None]  
```
---

### 7. Energy-Weighted Center-of-Gravity ($x_{cog}, y_{cog}, z_{cog}$)

**Importance:**  
Intermediate values used to compute r_std, z_std, and r90. Summarize the weighted “center” of the energy deposition in each event.  

**Mathematical Formulation:**  
- $x_{cog} = \frac{Σ_j E_j x_j}{E_{sum}}$
- $y_{cog} = \frac{Σ_j E_j y_j}{E_{sum}}$
- $z_{cog} = \frac{Σ_j E_j z_j}{E_{sum}}$

**Code Snippet:**  
```python
x_cog = cp.bincount(event_ids, weights=x*E, minlength=batch_end - batch_start)/E_sum  
y_cog = cp.bincount(event_ids, weights=y*E, minlength=batch_end - batch_start)/E_sum  
z_cog = cp.bincount(event_ids, weights=z*E, minlength=batch_end - batch_start)/E_sum  
```
---

## Summary Table

| Feature | Importance | Formula | Code Location |
|---------|------------|--------|---------------|
| E_sum | Total event energy | $Σ E_j$ | cp.bincount(event_ids, weights=E) |
| E_max | Maximum single hit energy | $\max(E_j)$ | cp.maximum.at(E_max, event_ids, E) |
| r_std | Transverse spread | $\sqrt{\frac{Σ_j E_j r_j^2}{E_{sum}} - (\frac{Σ_j E_j r_j}{E_{sum}})^2}$ | weighted_std(r) |
| z_std | Longitudinal spread | $\sqrt{(\frac{Σ_j E_j (z_j - z_{cog})^2}{E_{sum}} - (\frac{Σ_j E_j (z_j - z_{cog})}{E_{sum}})^2)}$ | weighted_std(z_shift) |
| r90 | 90% energy radius | $r_k$ where $Σ_{j≤k} E_j ≥ 0.9 E_{sum}$ | for-loop with cp.cumsum(E_sorted) |
| E_layer_frac[i] | Energy fraction per layer | $\frac{Σ_{layer-i} E_j}{E_{sum}}$ | cp.bincount(linear_idx, weights=E) / E_sum[:, None] |

---
