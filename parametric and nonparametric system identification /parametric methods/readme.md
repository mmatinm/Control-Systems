
# Parametric System Identification

This directory contains implementations of parametric system identification methods based on Ljung's "System Identification: Theory for the User." We applied these methods to a dataset from an unknown system, utilizing two types of inputs: Gaussian noise and PRBS (Pseudo-Random Binary Sequence). The selected configuration for the non-parametric identification was a gamma of 800 with a Hamming window.

## Preprocessing

The `preprocessing.m` file handles the following tasks:

- **Data Preprocessing**: Selecting Gaussian data for identification and PRBS for testing, as Gaussian data has higher energy and more autocorrelation.
- **Noise Characteristic Estimation**: Identifying the presence of colored noise (low-pass) in the system.
- **Delay Estimation**: Estimating the system delay using impulse response.
- **Order Estimation**: Using the N4SID method for order estimation.

From this point forward, we implement the following procedures for parametric system identification:

1. **Order Estimation**: For each method, we use AIC (Akaike Information Criterion), FPE (Final Prediction Error), and FIT (Fit Percentage).
2. **Residual Analysis**: Conducting a Residual Test on the test data (PRBS).
3. **Model Reduction**: Testing to see if the fitted model can be reduced.

In each file, we have plotted AIC, FPE, FIT percentages alongside zero-pole plots, residual tests, and simulated comparisons. However, for simplicity, not all plots are included in this README.

## ARX Model (`arx.m`)

In this file, we implemented the following methods for parametric identification using Gaussian data:

- **LS (Least Squares)**
- **IV (Instrumental Variable)**
- **RLS (Recursive Least Squares)**
- **RIV (Recursive Instrumental Variable)**

Although recursive methods do not have significant meaning in the context of the ARX model due to its linear regressor, we included them for learning purposes. We then applied AIC, FPE, and FIT methods alongside the Residual Test and model reduction on PRBS data to evaluate identification performance.

**Results**: The IV method yielded significantly better results than the others. Below are the simulated response comparisons and the fit percentages of different ARX models fitted by the IV method.

![ARX Response Comparison](images/arx_response_comparison.png)
![ARX Fit Percentages](images/arx_fit_percentages.png)

## ARMAX Model (`armax.m`)

In this file, we implemented the following methods for parametric identification using Gaussian data:

- **GN (Gauss-Newton)**
- **RML (Recursive Maximum Likelihood)**
- **RPLR (Recursive Prediction Error Method)**

We applied AIC, FPE, and FIT methods alongside the Residual Test and model reduction on PRBS data to evaluate identification performance.

**Results**: The GN method provided significantly better results than the others. Below are the simulated response comparisons and a description of the best-fitted ARMAX model by the GN method.

![ARMAX Response Comparison](images/armax_response_comparison.png)
![ARMAX Best Fitted Model](images/armax_best_fitted_model.png)

## Output Error Model (`OE.m`)

In this file, we implemented the following methods for parametric identification using Gaussian data by the Output Error (OE) model:

- **GN (Gauss-Newton)**
- **RLS (Recursive Least Squares)**
- **RIV (Recursive Instrumental Variable)**

We applied AIC, FPE, and FIT methods alongside the Residual Test and model reduction on PRBS data to evaluate identification performance.

**Results**: The GN method yielded significantly better results than the others. Below are the simulated response comparisons and a description of the best-fitted OE model by the GN method.

![OE Response Comparison](images/oe_response_comparison.png)
![OE Best Fitted Model](images/oe_best_fitted_model.png)

## Box-Jenkins Model (`BJ.m`)

In this file, we implemented the following methods for parametric identification using Gaussian data by the Box-Jenkins (BJ) model:

- **GN (Gauss-Newton)**
- **RPLR (Recursive Prediction Error Method)**

We applied AIC, FPE, and FIT methods alongside the Residual Test and model reduction on PRBS data to evaluate identification performance.

**Results**: The GN method provided significantly better results than the others. Below are the simulated response comparisons and a description of the best-fitted BJ model by the GN method.

![BJ Response Comparison](images/bj_response_comparison.png)
![BJ Best Fitted Model](images/bj_best_fitted_model.png)

## General Model (`General.m`)

In this file, we implemented the GN (Gauss-Newton) method for parametric identification using Gaussian data by a general model. We applied AIC, FPE, and FIT methods alongside the Residual Test and model reduction on PRBS data to evaluate identification performance.

**Results**: The GN method yielded significantly better results than the others. Below are the simulated response comparisons and a description of the best-fitted General model by the GN method.

![General Response Comparison](images/general_response_comparison.png)
![General Best Fitted Model](images/general_best_fitted_model.png)

## Conclusion

For the given system, the fitted ARMAX model was determined to be the best fit. Below are the residual tests for the best system, along with the uncertainty matrix of parameters using the `getcov` command in MATLAB and the Bode plot of the best model.

![Residual Test](images/residual_test.png)
![Uncertainty Matrix](images/uncertainty_matrix.png)
![Bode Plot](images/bode_plot.png)







---
we implemented parametric identification on a dataset from an unkown system based on Ljung's System Identification: Theory for the User
two inputs gaussian and PRBS 
the selected configuration from nonparametric identification.  gamma 800 with hamming window

preprocessing.m file
data preprocessing
selecting gaussian data for identification and prbs for test. because gaussian has more energy in a sense and more auto correlation
finding noise chracteristic ( in this system we have a colored noise (low pass))
delay estimation by impulse response
order estimation by n4sid

from this point forward we implement the following presiture for parametric system identification:
1. order estimation for each method by AIC and FPE and FIT 
2. implement and analysis of Residue test on test data (PRBS)
3. reducing the model order and test to see if the fitted model can be reduced 
in each file we have plotted AIC , FPE , FIT percent alongside zero pole plot , residual test and simulated comparison but we did not provide all of them for simplicity of read me file to read.

arx.m file
in this file we implemented LS(least square), IV (instrument variable), RLS(Recursive least square) and RIV methods ( although recursive methods doesnt have meaning in case of arx model because arx has linear regressor but we used this for learning purpose) to parametrically identification of the system using gaussian data. then we used AIC , FPE and FIT method alongside residue test and  model reduction on PRBS data to test the identification performance.
as a result we found out that using IV method can land significantly better results than others. below you can see the simulated response comparsion. and fit percent of different arx models fitted by IV method

image
image

armax.m file 
in this file we implemented GN(gauss newton), RML(recursive maximum likelyhood) and RPLR to parametrically identification of the system using gaussian data. then we used AIC , FPE and FIT method alongside residue test and  model reduction on PRBS data to test the identification performance.
as a result we found out that using GN method can land significantly better results than others. below you can see the simulated response comparsion. and description of the best fitted model of armax by GN

image
image

OE.m
in this file we implemented GN(gauss newton), RLS and RIV to parametrically identification of the system using gaussian data by OE (output error) model. then we used AIC , FPE and FIT method alongside residue test and  model reduction on PRBS data to test the identification performance.
as a result we found out that using GN method can land significantly better results than others. below you can see the simulated response comparsion. and description of the best fitted model of OE by GN

image

image

BJ.m
in this file we implemented GN(gauss newton) and RPLR to parametrically identification of the system using gaussian data by BJ (Box and Jeinciens) model. then we used AIC , FPE and FIT method alongside residue test and  model reduction on PRBS data to test the identification performance.
as a result we found out that using GN method can land significantly better results than others. below you can see the simulated response comparsion. and description of the best fitted model of BJ by GN

image
image

Genearl.m
in this file we implemented GN(gauss newton) to parametrically identification of the system using gaussian data by a general model. then we used AIC , FPE and FIT method alongside residue test and  model reduction on PRBS data to test the identification performance.
as a result we found out that using GN method can land significantly better results than others. below you can see the simulated response comparsion. and description of the best fitted model of General by GN

image
image


at last for the given system the fitted and metiond armax system was the best fit. here you can see its residual test for the best system along with uncertainity matrix of parameters using getcov command in matlab and bode plot ofthe best model.

image 
image 
image

