# HGCAL E_sum Distribution Analysis for Target Energy

## 1. Introduction

This analysis examines the **total deposited energy (E_sum)** in HGCAL calorimeter events corresponding to a specific **target true energy**. Understanding the distribution of E_sum allows assessment of the calorimeter response and energy resolution for events near the chosen target energy.

---

## 2. Methodology

- The processed dataset contains event-wise features including:
  - Total deposited energy (`E_sum`)
  - Maximum hit energy (`E_max`)
  - Radial and longitudinal spread (`r_std`, `z_std`)
  - Radius containing 90% of energy (`r90`)
  - Energy fraction per layer (`E_layer_frac`)

- For this analysis, events were selected within a **±0.5 GeV window** around the target energy of **150 GeV**.

- The **E_sum values** for these selected events were extracted, and a **histogram** was created to visualize the distribution.

- A **Gaussian fit** was applied to quantify the mean and standard deviation of the E_sum distribution, providing an estimate of the calorimeter’s energy resolution for this energy.

---

## 3. Results

The figure below shows the **E_sum histogram** for events near the target energy, along with the **Gaussian fit**:

![E_sum Distribution](./figures/e_sum_target_150GeV.png)

**Observations:**

- The E_sum distribution is approximately Gaussian, as expected for calorimeter energy response at a fixed incident energy.
- The Gaussian fit yields:
  - Mean (μ) ≈ value of μ from fit
  - Standard deviation (σ) ≈ value of σ from fit

- The width of the distribution (σ) reflects the **energy resolution** of the calorimeter at this energy.

---

## 4. Conclusion

- The total deposited energy (E_sum) for events near 150 GeV shows a well-defined Gaussian-like distribution.
- Gaussian fitting provides quantitative measures of mean energy deposition and its spread.
- This analysis can be extended to other target energies to study the **calorimeter response curve** and **energy resolution dependence** across the energy spectrum.

---

## 5. Notes

- The histogram is normalized to allow direct comparison with the Gaussian probability density function.
- The ±0.5 GeV tolerance window ensures that only events very close to the target energy are included, reducing contamination from other energy ranges.
- Ensure the figure file (`e_sum_target_150GeV.png`) is saved in the `figures/` directory for inclusion in the report.
