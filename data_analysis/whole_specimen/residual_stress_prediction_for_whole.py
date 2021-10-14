import numpy as np
from numpy.lib.function_base import average
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.decomposition import PCA

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,0.4])
  plt.legend()
  plt.show()

def data_import(csv_file_name):
    '''
    used to import dataset and split training and test dataset
    csv_file_name is the dateset root
    reture traing and test datasets
    '''
    raw_data = pd.read_csv(csv_file_name, sep='\t')
    x = raw_data.iloc[:,0:7]
    y = raw_data.iloc[:,7]
    return x, y

# Import dataset
x, y = data_import('extracted_data/simulation_data_for_whole.txt')

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=1/7.0, random_state=0)
# X_train = x.iloc[:1753750, :]
# X_test = x.iloc[1753750:, :]
# Y_train = y.iloc[:1753750]
# Y_test = y.iloc[1753750:]

print(Y_train.shape)
# Normalization
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
scaled_train_X = scaler_X.fit_transform(X_train)
scaled_test_X = scaler_X.transform(X_test)
scaled_train_Y = scaler_Y.fit_transform(Y_train.values.reshape(-1,1))
# scaled_test_Y = scaler_Y.transform(Y_test)

pca = PCA(n_components='mle')
pca.fit(scaled_train_X)
train_X_pca = pca.transform(scaled_train_X)
test_X_pca = pca.transform(scaled_test_X)

print(train_X_pca)
def build_model(N_hidden_nodes, input_dim, N_outputs):
  model = keras.Sequential([
    keras.layers.Dense(N_hidden_nodes, activation=tf.nn.leaky_relu, input_shape=(input_dim,)),
    keras.layers.Dense(N_outputs,activation='linear')
  ])

  
  model.compile(loss='mse',
                optimizer=tf.keras.optimizers.RMSprop(0.001),
                metrics=['mse'])
  return model
#build model
model = build_model(200, 7, 1)
model.summary()

history = model.fit(scaled_train_X, scaled_train_Y, batch_size=1, epochs=200, 
                    verbose=1, validation_split=0.2)
predictdata = model.predict(scaled_test_X)
oringin_data = Y_test
predict_data = scaler_Y.inverse_transform(predictdata)
plot_history(history)
model.save('data_analysis\whole_specimen\ANN.h5')

model = build_model(200, 7, 1)
model.summary()
history_pca = model.fit(scaled_train_X, scaled_train_Y, batch_size=1, epochs=200, 
                    verbose=1, validation_split=0.2)
predictdata_pca = model.predict(scaled_test_X)
predict_data_pca = scaler_Y.inverse_transform(predictdata_pca)
plot_history(history_pca)
model.save('data_analysis\whole_specimen\ANN_with_PCA.h5')

# print(type(predict_data))
# for i in range(len(predict_data)):
#     x = np.arange(0,0.182,0.003)
#     plt.plot(x,oringin_data.iloc[i,:].ravel(),label = 'Target stress')
#     plt.plot(x,predict_data[i,:].ravel(),'r--',label = 'Predict stress')
#     plt.plot(x,predict_data_pca[i,:].ravel(),'g--',label = 'Predict stress with PCA')
#     plt.legend()
#     plt.show()
