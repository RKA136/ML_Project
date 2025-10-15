# HGCAL Measured Energy per Layer for Multiple Target Energies

## 1. Introduction

This analysis examines the **measured energy deposited per layer** in the HGCAL calorimeter for multiple **target true energies**. Studying the layer-wise energy deposition allows insight into the **longitudinal shower development** and **calorimeter response** at different incident energies.

---

## 2. Methodology

- The processed dataset contains per-event features including:
  - Total deposited energy (`E_sum`)
  - Maximum hit energy (`E_max`)
  - Radial and longitudinal spread (`r_std`, `z_std`)
  - Radius containing 90% of energy (`r90`)
  - Energy fraction per layer (`E_layer_frac`)

- Events were selected within a **±0.5 GeV window** around each target energy:
  - 20 GeV
  - 100 GeV
  - 300 GeV

- For the selected events, **absolute energy per layer** was computed by multiplying `E_sum` by the per-layer energy fraction.

- For each z layer, the **mean** and **standard deviation** of measured energy across events were calculated and visualized using **error bars**.

- Distinct colors were used for each target energy, and the standard deviation was represented as vertical error bars to highlight event-to-event fluctuations.

---

## 3. Results

The figure below presents the **measured energy per layer** for the three target energies:

![Measured Energy per Layer](./figures/measured_energy_per_layer.png)

**Observations:**

- Energy deposition profiles are roughly Gaussian-like along the longitudinal (z) direction, consistent with expected shower development in a calorimeter.
- Lower energy events (20 GeV) show smaller overall energy deposition and fewer hits in deeper layers.
- Higher energy events (100 and 300 GeV) penetrate more layers, with peak energy deposition occurring in intermediate layers.
- Standard deviations indicate **significant event-to-event fluctuations**, which are expected in particle showers.

---

## 4. Conclusion

- Layer-wise analysis across multiple target energies highlights the **energy-dependent longitudinal shower profile** in the HGCAL calorimeter.
- Error bars demonstrate the **natural fluctuations in energy deposition** per layer.
- This visualization framework can be extended to study correlations between layer-wise energy deposition and other features such as radial spread (`r_std`) or total deposited energy (`E_sum`) for a comprehensive calorimeter response analysis.

---