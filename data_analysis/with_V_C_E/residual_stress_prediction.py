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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
def get_plot(oringin_data, predict_data, predict_data_pca, i):
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(-90, 91,3)
    oringin_data = oringin_data / 1000000
    predict_data = predict_data / 1000000
    predict_data_pca = predict_data_pca / 1000000
    p1 = ax.plot(x,oringin_data.ravel(),'r--', label = 'Target stress')
    p2 = ax.plot(x,predict_data.ravel(),'g--',label = 'Predict stress')
    p3 = ax.plot(x,predict_data_pca.ravel(),'b--',label = 'Predict stress with PCA')
    ax.set_title("Test-Set" + str(i))
    # ax.set_xticks(x)
    ax.set_ylabel('Logitudinal stress (MPa)')
    ax.set_xlabel('Distance from specimen mid-length X(mm)')
    ax.yaxis.set_tick_params(direction='out')

    ax2 = plt.twinx()
    difference = predict_data.ravel() - oringin_data.ravel()
    # difference=np.array(list(map(lambda x,y:x/y,difference,oringin_data.ravel())))
    difference_with_pca = predict_data_pca.ravel() - oringin_data.ravel()
    # difference_with_pca=np.array(list(map(lambda x,y:x/y,difference_with_pca,oringin_data.ravel())))
    ymax=np.max([difference.max(),difference_with_pca.max()])*(1+0.1)
    rects1=ax2.bar(x-1/2.0, difference, 1, color="tab:brown")
    rects2=ax2.bar(x+1/2.0, difference_with_pca, 1, color="tab:red")
    ax2.set_ylabel('Error (MPa)')

    legend_elements = [Line2D([0], [0], color='red', lw=2, label='Benchmark'),
                       Line2D([0], [0], color='green', lw=2, label='ANN prediction'),
                       Line2D([0], [0], color='blue', lw=2, label='ANN (PCA) prediction'),
                       Patch(facecolor='tab:brown', edgecolor='b',label='Error'),
                       Patch(facecolor='tab:red', edgecolor='b',label='Error (PCA)'),]

    ax.legend(handles=legend_elements, loc='best')
    plt.savefig('H:/PhD_git/ANN_results/with_V_C_E/test-' + str(i) + '.jpg')
    plt.close()

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
    raw_data = pd.read_csv(csv_file_name).dropna()
    x = raw_data.iloc[:,:6]
    y = raw_data.iloc[:,8:]
    return x, y

# Import dataset
x, y = data_import(r'extracted_data/simulation_data_with_V_C_E.csv')

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=1/10.0, random_state=0)
# Normalization
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
scaled_train_X = scaler_X.fit_transform(X_train)
scaled_test_X = scaler_X.transform(X_test)
scaled_train_Y = scaler_Y.fit_transform(Y_train)
# scaled_test_Y = scaler_Y.transform(Y_test)

pca = PCA(n_components='mle')
pca.fit(scaled_train_X)
train_X_pca = pca.transform(scaled_train_X)
test_X_pca = pca.transform(scaled_test_X)


def build_model(N_hidden_nodes, input_dim, N_outputs):
  model = keras.Sequential([
    keras.layers.Dense(N_hidden_nodes, activation=tf.nn.leaky_relu, input_shape=(input_dim,)),
    keras.layers.Dense(N_outputs,activation='linear')
  ])

  
  model.compile(loss='mse',
                optimizer=tf.keras.optimizers.RMSprop(0.001),
                metrics=['mse'])
  return model

model = build_model(200, 6, 61)
model.summary()

history = model.fit(scaled_train_X, scaled_train_Y, batch_size=1, epochs=200, 
                    verbose=1, validation_split=0.2)
predictdata = model.predict(scaled_test_X)
oringin_data = Y_test
predict_data = scaler_Y.inverse_transform(predictdata)
# plot_history(history)

model_pca = build_model(200, 6, 61)
model_pca.summary()
history_pca = model_pca.fit(scaled_train_X, scaled_train_Y, batch_size=1, epochs=200, 
                    verbose=1, validation_split=0.2)
predictdata_pca = model_pca.predict(scaled_test_X)
predict_data_pca = scaler_Y.inverse_transform(predictdata_pca)
# plot_history(history_pca)

for i in range(len(predict_data)):
  get_plot(oringin_data.iloc[i,:], predict_data[i,:], predict_data_pca[i,:], i)


# x_benchmark, y_benchmark = data_import('H:/PhD_data/CSF_results/extracted_data/benchmark.csv')
# scaled_benchmark_X = scaler_X.transform(x_benchmark)
# predictdata_benchmark = model_pca.predict(scaled_benchmark_X)
# predict_data_benchmark = scaler_Y.inverse_transform(predictdata_benchmark)
# x = np.arange(0,0.182,0.003)
# plt.plot(x,y_benchmark.values.ravel(),label = 'benchmark stress')
# plt.plot(x,predict_data_benchmark[:,:].ravel(),'r--',label = 'Predict stress')
# plt.title("Benchmark")
# plt.legend()
# plt.show()