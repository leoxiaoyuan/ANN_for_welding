import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

csv_file_name = 'H:\PhD_data\CSF_results\extracted_data\simulation_data.csv'
raw_data = pd.read_csv(csv_file_name).dropna()
test_index = 14
train_index = np.delete(np.arange(raw_data.shape[0]), test_index)
X_train = raw_data.iloc[train_index,0:4]
X_test = raw_data.iloc[test_index,0:4]
Y_train = raw_data.iloc[train_index,4:6]
Y_test = raw_data.iloc[test_index,4:6]

NB_EPOCH = 200
BATCH_SIZE = 1
VERBOSE = 1
NB_CLASSES = 2
OPTIMIZER = SGD()
N_HIDDEN = 10
VALICATION_SPLIT = 0.2

scaler_X = StandardScaler()
scaler_Y = StandardScaler()
scaled_train_X = scaler_X.fit_transform(X_train.values.reshape(-1,4))
scaled_test_X = scaler_X.transform(X_test.values.reshape(-1,4))
scaled_train_Y = scaler_Y.fit_transform(Y_train.values.reshape(-1,2))
scaled_test_Y = scaler_Y.transform(Y_test.values.reshape(-1,2))

input_dim = 4
input_layer = Input(shape=(input_dim,))
encoded = input_layer

encoded = Dense(N_HIDDEN, activation='relu')(encoded)

encoded = Dense(NB_CLASSES, activation='linear')(encoded)

model = Model(inputs=input_layer, outputs=encoded)
model.summary()

model.compile(loss='mse', optimizer=OPTIMIZER, metrics=['mse'])
history = model.fit(scaled_train_X, scaled_train_Y, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALICATION_SPLIT)
predictdata = model.predict(scaled_test_X)
oringin_data = scaler_Y.inverse_transform(scaled_test_Y)
predict_data = scaler_Y.inverse_transform(predictdata)

plt.scatter('bead_width',oringin_data.ravel()[0],marker = '+', color = 'blue')
plt.scatter('bead_depth',oringin_data.ravel()[1],marker = 'o', color = 'blue')
plt.scatter('bead_width',predict_data.ravel()[0],marker = '+', color = 'green')
plt.scatter('bead_depth',predict_data.ravel()[1],marker = 'o', color = 'green')


plt.show()