
# Parametric System Identification

This directory contains implementations of parametric system identification methods based on Ljung's "System Identification: Theory for the User." We applied these methods to a dataset from an unknown system, utilizing two types of inputs: Gaussian noise and PRBS (Pseudo-Random Binary Sequence). The selected configuration for the non-parametric identification was a gamma of 800 with a Hamming window.

---
## Table of Contents
- [Preprocessing](#Preprocessing)
- [ARX-Model](#ARX-Model)
- [ARMAX-Model](#ARMAX-Model)
- [Output-Error-Model](#Output-Error-Model)
- [Box-Jenkins-Model](#Box-Jenkins-Model)
- [General-Model](#general-model)
- [Conclusion](#conclusion)
---

## Preprocessing

The `Preprocessing.m` file handles the following tasks:

- **Data Preprocessing**: Selecting Gaussian data for identification and PRBS for testing, as Gaussian data has higher energy and more autocorrelation at zero lag.
- **Noise Characteristic Estimation**: Identifying the Noise Spectrum, here we have colored noise (low-pass) in the system.
- **Delay Estimation**: Estimating the system delay using impulse response.
- **Order Estimation**: Using the N4SID method for order estimation.

From this point forward, we implement the following procedures for parametric system identification:

1. **Order Estimation**: For each method, we use AIC (Akaike Information Criterion), FPE (Final Prediction Error), and FIT (Fit Percentage).
2. **Residual Analysis**: Conducting a Residual Test on the test data (PRBS).
3. **Model Reduction**: Reducing model orders to check for over-parameterization and model simplification

In each file, we have plotted AIC, FPE, FIT percentages alongside zero-pole plots, residual tests, and simulated comparisons. However, for simplicity, not all plots are included in this README.

---
## ARX Model

In `ARX.m` file, we implemented the following methods for parametric identification using Gaussian data:

- **LS (Least Squares)**
- **IV (Instrumental Variable)**
- **RLS (Recursive Least Squares)**
- **RIV (Recursive Instrumental Variable)**

Although recursive methods do not have significant meaning in the context of the ARX model due to its linear regressor, we included them for learning purposes. We then applied AIC, FPE, and FIT methods alongside the Residual Test and model reduction on PRBS data to evaluate identification performance.

**Results**: The IV method yielded significantly better results than the others. Below are the simulated response comparisons and the fit percentages of different ARX models fitted by the IV method.

![ARX Response Comparison](images/arx_response_comparison.png)
![ARX Fit Percentages](images/arx_fit_percentages.png)

---
## ARMAX Model 

In `ARMAX.m` file, we implemented the following methods for parametric identification using Gaussian data:

- **GN (Gauss-Newton)**
- **RML (Recursive Maximum Likelihood)**
- **RPLR (Recursive Prediction Error Method)**

We applied AIC, FPE, and FIT methods alongside the Residual Test and model reduction on PRBS data to evaluate identification performance.

**Results**: The GN method provided significantly better results than the others. Below are the simulated response comparisons and a description of the best-fitted ARMAX model by the GN method.

![ARMAX Response Comparison](images/armax_response_comparison.png)
![ARMAX Best Fitted Model](images/armax_best_fitted_model.png)

## Output Error Model 

In `OE.m` file, we implemented the following methods for parametric identification using Gaussian data by the Output Error (OE) model:

- **GN (Gauss-Newton)**
- **RLS (Recursive Least Squares)**
- **RIV (Recursive Instrumental Variable)**

We applied AIC, FPE, and FIT methods alongside the Residual Test and model reduction on PRBS data to evaluate identification performance.

**Results**: The GN method yielded significantly better results than the others. Below are the simulated response comparisons and a description of the best-fitted OE model by the GN method.

![OE Response Comparison](images/oe_response_comparison.png)
![OE Best Fitted Model](images/oe_best_fitted_model.png)

---
## Box-Jenkins Model 

In `BJ.m` file, we implemented the following methods for parametric identification using Gaussian data by the Box-Jenkins (BJ) model:

- **GN (Gauss-Newton)**
- **RPLR (Recursive Prediction Error Method)**

We applied AIC, FPE, and FIT methods alongside the Residual Test and model reduction on PRBS data to evaluate identification performance.

**Results**: The GN method provided significantly better results than the others. Below are the simulated response comparisons and a description of the best-fitted BJ model by the GN method.

![BJ Response Comparison](images/bj_response_comparison.png)
![BJ Best Fitted Model](images/bj_best_fitted_model.png)

---
## General Model 

In `General.m` file, we implemented the GN (Gauss-Newton) method for parametric identification using Gaussian data by a general model. We applied AIC, FPE, and FIT methods alongside the Residual Test and model reduction on PRBS data to evaluate identification performance.

**Results**: The GN method yielded significantly better results than the others. Below are the simulated response comparisons and a description of the best-fitted General model by the GN method.

![General Response Comparison](images/general_response_comparison.png)
![General Best Fitted Model](images/general_best_fitted_model.png)

---
## Conclusion

For the given system, the fitted ARMAX model was determined to be the best fit. Below are the residual tests for the best system, along with the uncertainty matrix of parameters using the `getcov` command in MATLAB and the Bode plot of the best model.

![Residual Test](images/residual_test.png)
![Uncertainty Matrix](images/uncertainty_matrix.png)
![Bode Plot](images/bode_plot.png)


