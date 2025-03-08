'''
if needed
pip install control  
'''

import numpy as np
import pandas as pd
import control
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import StandardScaler
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from scipy import signal
import tensorflow as tf
from scipy.signal import sawtooth


###########################  Generate data
'''
in this section we simulated a mass-spring-damper system in response to a square signal as force. 
then we organised this force and response data to make train, test and validation data for training the model.
'''
m = 24
c = 110
k = 2500

#train data
ttr = np.linspace(0, 10, 500)
forcetr = 4*(signal.square(2 * np.pi * 0.2 * ttr))

systemtr = control.TransferFunction([1], [m, c, k])
ttr, positiontr = control.forced_response(systemtr, T=ttr, U=forcetr)
velocitytr = np.gradient(positiontr, ttr)
accelerationtr = np.gradient(velocitytr, ttr)

scaler_Xtr = StandardScaler()
scaler_ytr = StandardScaler()

forcetr_normalized = scaler_Xtr.fit_transform(forcetr.reshape(-1, 1))
positiontr_normalized = scaler_ytr.fit_transform(positiontr.reshape(-1, 1))

X_train = np.column_stack((forcetr_normalized, np.zeros((forcetr_normalized.shape[0],1))))
y_train = positiontr_normalized

#validation data
tv = np.linspace(0, 10, 500)
forcev = 5*(signal.square(2 * np.pi * 0.2 * tv))

systemv = control.TransferFunction([1], [m, c, k])
tv, positionv = control.forced_response(systemv, T=tv, U=forcev)
velocityv = np.gradient(positionv, tv)
accelerationv = np.gradient(velocityv, tv)

scaler_Xv = StandardScaler()
scaler_yv = StandardScaler()

forcev_normalized = scaler_Xv.fit_transform(forcev.reshape(-1, 1))
positionv_normalized = scaler_yv.fit_transform(positionv.reshape(-1, 1))

X_val = np.column_stack((forcev_normalized, np.zeros((forcev_normalized.shape[0],1))))
y_val = positionv_normalized

#test data
tts = np.linspace(0, 10, 500)
forcets = 6*(signal.square(2 * np.pi * 0.2 * tts))

systemts = control.TransferFunction([1], [m, c, k])
tts, positionts = control.forced_response(systemts, T=tts, U=forcets)
velocityts = np.gradient(positionts, tts)
accelerationts = np.gradient(velocityts, tts)

scaler_Xts = StandardScaler()
scaler_yts = StandardScaler()

forcets_normalized = scaler_Xts.fit_transform(forcets.reshape(-1, 1))
positionts_normalized = scaler_yts.fit_transform(positionts.reshape(-1, 1))

X_test = np.column_stack((forcets_normalized, np.zeros((forcets_normalized.shape[0],1))))
y_test = positionts_normalized

'''
# you can use this code to plot train, test and validation data
def fourplot(a, X_train, y_train, accelerationtr, velocitytr):
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.scatter(a, X_train[:,0])
    #plt.plot(t, pos)
    plt.ylabel('force scaled train')
    plt.title('Mass-Spring-Damper System Response')

    plt.subplot(4, 1, 2)
    plt.scatter(a, y_train)
    #plt.plot(t, vel)
    plt.ylabel('Position')

    plt.subplot(4, 1, 3)
    plt.scatter(a, velocitytr)
    #plt.plot(t, acc)
    plt.ylabel('Velocity ')
    plt.xlabel('Time (s)')

    plt.subplot(4, 1, 4)
    plt.scatter(a, accelerationtr)
    #plt.plot(t, acc)
    plt.ylabel('Acceleration ')
    plt.xlabel('Time (s)')

    plt.show()

fourplot(ttr, X_train, y_train, accelerationtr, velocitytr)
fourplot(tv, X_val, y_val, accelerationv, velocityv)
fourplot(tts, X_test, y_test, accelerationts, velocityts)
'''

########################### pre processing data
'''
the most important part of this project was to determine how should we determine input and ouput of our network. 
'''
X_train = X_train.reshape((1,X_train.shape[0],X_train.shape[1]))
y_train = y_train.reshape((1,y_train.shape[0],1))

X_val = X_val.reshape((1,X_val.shape[0],X_val.shape[1]))
y_val = y_val.reshape((1,y_val.shape[0],1))

X_test = X_test.reshape((1,X_test.shape[0],X_test.shape[1]))
y_test = y_test.reshape((1,y_test.shape[0],1))

########################### model
model=Sequential()
model.add(SimpleRNN(6, input_shape=(X_train.shape[1],X_train.shape[2]), activation='linear',return_sequences=True))  #use_bias=False
model.add(Dense(1, activation='linear', use_bias=False))
model.compile(optimizer=RMSprop(learning_rate=0.001 , rho=0.9), loss='mean_squared_error')

checkpoint_path = 'model_checkpoint.h5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch')
callbacks_list = [checkpoint]
validation_loss_list = []
model_weights_list = []

epochs = 1000
for epoch in range(epochs):
    history = model.fit(X_train, y_train, batch_size=1,callbacks=callbacks_list ,validation_data=(X_val, y_val), verbose=2)
    validation_loss_list.append(history.history['val_loss'])
    model_weights_list.append(model.get_weights())

model.compile(optimizer=RMSprop(learning_rate=0.0001 , rho=0.9), loss='mean_squared_error')
epochs = 100
for epoch in range(epochs):
    history = model.fit(X_train, y_train, batch_size=1,callbacks=callbacks_list ,validation_data=(X_val, y_val), verbose=2)
    validation_loss_list.append(history.history['val_loss'])
    model_weights_list.append(model.get_weights())

#model.save('/content/new_drive/MyDrive/trained_SRNN.keras')
#trainedmodel = tf.keras.models.load_model('/content/new_drive/MyDrive/trained_SRNN.keras')

########################### test with a data set like the train data
train_predictions = trainedmodel.predict(X_train)
test_predictions = trainedmodel.predict(X_test)
pred = scaler_yts.inverse_transform(test_predictions.reshape((test_predictions.shape[0], test_predictions.shape[1])))
pred = pred.reshape(pred.shape[1],1)
tpred = scaler_ytr.inverse_transform(train_predictions.reshape((train_predictions.shape[0], train_predictions.shape[1])))
tpred = tpred.reshape(tpred.shape[1],1)

ACCpredicted = np.zeros((len(pred),1))
VELpredicted = np.zeros((len(pred),1))
for i in range(1, len(pred)):
    VELpredicted[i] = (pred[i] - pred[i-1]) /0.02
for i in range(1, len(pred)):
    ACCpredicted[i] = (VELpredicted[i] - VELpredicted[i-1]) /0.02
'''
velocitytr = velocitytr.reshape(velocitytr.shape[1],1)
velocityts = velocityts.reshape(velocityts.shape[1],1)
positionts = positionts.reshape(positionts.shape[1],1)
accelerationts = accelerationts.reshape(accelerationts.shape[1],1)

tts = tts.reshape(-1,1)
ttr = ttr.reshape(-1,1)
'''
# Plot the results
plt.figure(figsize=(6, 7))
plt.subplot(4, 1, 1)
plt.plot(tts, forcets, c = 'orange')
plt.title('Square Signal Response', size = 12)
#plt.xlabel('Time (seconds)')
plt.ylabel('Force (N)', size = 12)

plt.subplot(4, 1, 2)
plt.plot(tts,positionts, label='Real values', c = 'orange')
plt.scatter(tts,pred, label='Predictions', s = 3, c = 'blue')
plt.legend()
#plt.xlabel('Time (seconds)')
plt.ylabel('Pos (m)', size = 12)
#plt.title('Predictions')

plt.subplot(4, 1, 3)
plt.plot(tts,velocityts, label='Real Velocity', c = 'orange')
plt.scatter(tts,VELpredicted, label='Predictions', s = 3, c = 'blue')
#plt.legend()
#plt.xlabel('Time')
plt.ylabel('Vel (m/s)', size = 12)


plt.subplot(4, 1, 4)
plt.plot(tts,accelerationts, label='Real Acceleration', c = 'orange')
plt.scatter(tts,ACCpredicted, label='Predictions', s = 3, c = 'blue')
#plt.legend()
plt.xlabel('Time (seconds)', size = 12)
plt.ylabel('Acc (m/s^2)', size = 12)

#plt.savefig('/content/new_drive/MyDrive/square.testpout.png')
plt.show()


########################### sawtooth force input test
m = 24
c = 110
k = 2500

system1 = control.TransferFunction([1], [m, c, k])
t1 = np.linspace(0, 10, 500)
force1 = 6*(signal.sawtooth(2 * np.pi * 0.2 * t1))
t1, position1 = control.forced_response(system1, T=t1, U=force1)
velocity1 = np.gradient(position1, t1)
acceleration1 = np.gradient(velocity1, t1)

t1 = t1.reshape(-1, 1)
position1 = position1.reshape(-1,1)
velocity1 = velocity1.reshape(-1,1)
acceleration1 =acceleration1.reshape(-1,1)

scaler_X1 = StandardScaler()
scaler_y1 = StandardScaler()

force1_normalized = scaler_X1.fit_transform(force1.reshape(-1,1))
position1_normalized = scaler_y1.fit_transform(position1)

X_test1 = np.column_stack((force1_normalized, np.zeros((force1_normalized.shape[0],1))))
y_test1 = position1_normalized
X_test1 = X_test1.reshape((1,X_test1.shape[0],X_test1.shape[1]))

pred1 = trainedmodel.predict(X_test1)
pred1 = scaler_y1.inverse_transform(pred1.reshape((pred1.shape[0], pred1.shape[1])))   #finalmodel1
pred1 = pred1.reshape(-1,1)

ACCpredicted1 = np.zeros((len(pred1),1))
VELpredicted1 = np.zeros((len(pred1),1))
#POSpredicted1[0] = position1[1+sequence_length]
for i in range(1, len(pred1)):
    VELpredicted1[i] = (pred1[i] - pred1[i-1]) /0.02
for i in range(1, len(pred1)):
    ACCpredicted1[i] = (VELpredicted1[i] - VELpredicted1[i-1]) /0.02

# Plot the results
plt.figure(figsize=(6, 7))
plt.subplot(4, 1, 1)
plt.plot(t1, force1 , c = 'orange')
plt.title('Sawtooth Signal Response', size = 12)
#plt.xlabel('Time (seconds)')
plt.ylabel('Force (N)', size = 12)

plt.subplot(4, 1, 2)
plt.plot(t1,position1, label='Real values' , c = 'orange')
plt.scatter(t1,pred1, label='Predictions', s = 3, c = 'blue')
plt.legend()
#plt.xlabel('Time')
plt.ylabel('Pos (m)', size = 12)
#plt.title('Predictions')

plt.subplot(4, 1, 3)
plt.plot(t1,velocity1, label='Real Velocity' , c = 'orange')
plt.scatter(t1,VELpredicted1, label='Predictions', s = 3, c = 'blue')
#plt.legend()
#plt.xlabel('Time')
plt.ylabel('Vel (m/s)', size = 12)

plt.subplot(4, 1, 4)
plt.plot(t1,acceleration1, label='Real Acceleration', c = 'orange')
plt.scatter(t1,ACCpredicted1, label='Predictions', s = 3, c = 'blue')
#plt.legend()
plt.xlabel('Time(seconds)', size = 12)
plt.ylabel('Acc (m/s^2)', size = 12)

#plt.savefig('/content/new_drive/MyDrive/sawtooth.testpout.jpg')
plt.show()


########################### sin force input test
m = 24
c = 110
k = 2500

system2 = control.TransferFunction([1], [m, c, k])
t2 = np.linspace(0, 10, 500)
force2 = 6*(np.sin(2 * np.pi * 0.2 * t2))
t1, position2 = control.forced_response(system2, T=t2, U=force2)
velocity2 = np.gradient(position2, t2)
acceleration2 = np.gradient(velocity2, t2)

t2 = t2.reshape(-1, 1)
position2 = position2.reshape(-1,1)
velocity2 = velocity2.reshape(-1,1)
acceleration2 =acceleration2.reshape(-1,1)

scaler_X2 = StandardScaler()
scaler_y2 = StandardScaler()

force2_normalized = scaler_X2.fit_transform(force2.reshape(-1,1))
position2_normalized = scaler_y2.fit_transform(position2)

X_test2 = np.column_stack((force2_normalized, np.zeros((force2_normalized.shape[0],1))))
y_test2 = position2_normalized
X_test2 = X_test2.reshape((1,X_test2.shape[0],X_test2.shape[1]))

pred2 = trainedmodel.predict(X_test2)
pred2 = scaler_y2.inverse_transform(pred2.reshape((pred2.shape[0], pred2.shape[1])))   #finalmodel1
pred2 = pred2.reshape(-1,1)

ACCpredicted2 = np.zeros((len(pred2),1))
VELpredicted2 = np.zeros((len(pred2),1))
#POSpredicted1[0] = position1[1+sequence_length]
for i in range(1, len(pred2)):
    VELpredicted2[i] = (pred2[i] - pred2[i-1]) /0.02
for i in range(1, len(pred2)):
    ACCpredicted2[i] = (VELpredicted2[i] - VELpredicted2[i-1]) /0.02

# Plot the results
plt.figure(figsize=(6, 7))
plt.subplot(4, 1, 1)
plt.plot(t2, force2 , c = 'orange')
plt.title('Sinusoidal Signal Response', size = 16)
#plt.xlabel('Time (seconds)')
plt.ylabel('Force (N)', size = 12)

plt.subplot(4, 1, 2)
plt.plot(t2,position2, label='Real values', c = 'orange')
plt.scatter(t2,pred2, label='Predictions', s = 3, c = 'blue')
plt.legend()
#plt.xlabel('Time')
plt.ylabel('Pos (m)', size = 12)
#plt.title('Predictions')

plt.subplot(4, 1, 3)
plt.plot(t2,velocity2, label='Real Velocity', c='orange')
plt.scatter(t2,VELpredicted2, label='Predictions', s = 3, c = 'blue')
#plt.legend()
#plt.xlabel('Time')
plt.ylabel('Vel (m/s)', size = 12)

plt.subplot(4, 1, 4)
plt.plot(t2,acceleration2, label='Real Acceleration', c = 'orange')
plt.scatter(t2,ACCpredicted2, label='Predictions', s = 3, c = 'blue')
#plt.legend()
plt.xlabel('Time (seconds)', size = 12)
plt.ylabel('Acc (m/s^2)', size = 12)
#plt.savefig('/content/new_drive/MyDrive/sin.testpout.jpg')
plt.show()



