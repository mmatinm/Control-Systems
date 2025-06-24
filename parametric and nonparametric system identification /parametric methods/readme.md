
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

