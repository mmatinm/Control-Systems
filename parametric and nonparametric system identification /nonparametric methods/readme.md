# System Identification Using Simulated Data

This repository contains MATLAB code for a system identification project performed on simulated data. The project follows methodologies from Lennart Ljung's *System Identification: Theory for the User*, using both **nonparametric** and **parametric** modeling techniques.

## 📊 Dataset

The dataset consists of a single `.mat` file located in the `data/` directory and contains four columns:

- `u1` – Input signal (white noise)
- `y1` – Output corresponding to `u1`
- `u2` – Input signal (colored noise)
- `y2` – Output corresponding to `u2`

The true system that generated this data is unknown.

## Nonparametric Methods

These methods help estimate system characteristics directly from data without fitting a parametric model.

---

### Correlation Analysis

**Description:**  
Cross-correlation between input and output helps identify delays and linear dependence. Autocorrelation of residuals evaluates whether the noise is white (uncorrelated), important for assessing model assumptions.

**MATLAB File:** `data1.mat`

**Plots:**

- Cross-correlation:
  ![Cross-correlation](results/cross_correlation.png)

- Residual autocorrelation:
  ![Residual autocorrelation](results/residual_autocorrelation.png)

---

### ETFE (Empirical Transfer Function Estimate)

**Description:**  
ETFE provides a raw estimate of the frequency response by dividing the FFT of the output by the FFT of the input. It's fast but may be noisy due to lack of smoothing.

**MATLAB File:** `scripts/etfe_analysis.m`

**Plots:**

- ETFE magnitude:
  ![ETFE Magnitude](results/etfe_magnitude.png)

- ETFE phase:
  ![ETFE Phase](results/etfe_phase.png)

---

### Mean ETFE (Averaged ETFE)

**Description:**  
An improved version of ETFE that averages spectral estimates across multiple segments, reducing variance and improving readability of the frequency response.

**MATLAB File:** `scripts/mean_etfe_analysis.m`

**Plots:**

- Mean ETFE magnitude:
  ![Mean ETFE Magnitude](results/mean_etfe_magnitude.png)

- Mean ETFE phase:
  ![Mean ETFE Phase](results/mean_etfe_phase.png)

---

### SPA (Spectral Analysis)

**Description:**  
SPA uses smoothed spectral density estimates to compute the system's frequency response and includes confidence intervals, providing insight into uncertainty.

**MATLAB File:** `scripts/spa_analysis.m`

**Plots:**

- SPA Bode plot with confidence bounds:
  ![SPA Bode Plot](results/spa_bode.png)

- Noise spectrum estimate:
  ![Noise Spectrum](results/spa_noise_spectrum.png)

---

