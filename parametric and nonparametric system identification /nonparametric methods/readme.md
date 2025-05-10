# System Identification Using Simulated Data

This directory contains MATLAB code for a system identification project performed on simulated data. The project follows methodologies from Lennart Ljung's *System Identification: Theory for the User*, using both **nonparametric** and **parametric** modeling techniques.

## 📊 Dataset

The dataset consists of a single `.mat` file located in this directory and contains four columns:

- `uw` – Input signal
- `yw` – Output corresponding to `uw` (white noise)
- `uc` – Input signal 
- `yc` – Output corresponding to `uc` (colored noise)

The true system that generated this data is unknown.

**MATLAB File:** `data1.mat`


## Nonparametric Methods

These methods help estimate system characteristics directly from data without fitting a parametric model.

---

### Correlation Analysis

**Description:**  
Correlation analysis estimates the system's impulse response by computing the cross-correlation between the input and output signals. This approach assumes a linear time-invariant system and white input noise. The impulse response provides insight into the system's dynamics, such as delay and transient behavior.

The analysis also includes checking the autocorrelation of the residuals to determine if the remaining noise is uncorrelated (i.e., white). This is important for validating the assumptions of the model and identifying potential structure in the noise that the model fails to capture.

In this project, the effect of the averaging data size \( N \) on the correlation estimates was evaluated using four different values: **1024**, **2048**, **4096**, and **8192**. 

**Dataset Size:** 16,384 rows

**MATLAB File:** `S1correlation.m`

**Plots:**

- Evaluated impulse responses ( system with white noise ):
  ![corr2](images/corr2.jpg)

- Evaluated impulse responses ( system with colored noise ):
  ![corr3](images/corr3.jpg)

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

