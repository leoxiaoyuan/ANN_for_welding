from traceback import print_tb
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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
from bayes_opt import BayesianOptimization
import warnings
import os
import random
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, WhiteKernel, ConstantKernel as C, RationalQuadratic as R
warnings.filterwarnings('ignore')

def get_plot(oringin_data, predict_data, i):
    '''
    Plot target outputs against predicted outputs
    inputs:
        oringin_data: target outputs
        predict_data: predicted outputs
        i: plot index
        root: root for saving figure
    '''
    root = 'H:/PhD_git/ANN_results/BD/test/test--' + str(i) + '.jpg'
    fig, ax = plt.subplots(figsize=(10,6))
    x = y_label_new
    p1 = ax.plot(x,oringin_data.ravel(),'r--', label = 'Target stress')
    p2 = ax.plot(x,predict_data,'g--',label = 'Predict stress')
    ax.set_title("Test-Set" + str(i))
    # ax.set_xticks(x)
    ax.set_ylabel('Logitudinal stress (MPa)')
    ax.set_xlabel('Distance from top surface Z(mm)')
    
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_tick_params(direction='out')

    ax2 = plt.twinx()
    difference = predict_data - oringin_data.ravel()
    difference=np.array(list(map(lambda x,y:x/y,difference,oringin_data.ravel())))
    # rects1=ax2.bar(x-0.5, difference, 0.3, color="tab:brown")
    # ax2.set_ylabel('Percentage Error (%)') 


    legend_elements = [Line2D([0], [0], color='red', lw=2, label='Simulation result'),
                       Line2D([0], [0], color='green', lw=2, label='ANN prediction'),
                      #  Patch(facecolor='tab:brown', edgecolor='b',label='Percentage Error'),
                      ]

    ax.legend(handles=legend_elements, loc='best')
    plt.savefig(root)
    # if i == 'benchmark3':
    #   plt.savefig('H:/PhD_git/ANN_results/BD/test--' + 'benchmark3' + '.jpg')
    plt.close()

def plot_history(history):
    '''
    Plot learning curve using NN training history info
    '''
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
    raw_data = pd.read_csv(csv_file_name,header=None).dropna()
    x = raw_data.iloc[:,3:6]
    # x.columns = ['Travel_length', 'Welding_speed', 'Net_energy_input']
    y = raw_data.iloc[:,6:]
    return x, y

def nn_cl_bo(N_hidden_nodes, l_rate, Batch_size):
    '''
    Using certain number of neurons, learning rate and Batch size
    to train a NN and return its cross validation MSE
    inputs:
        N_hidden_nodes: number of neurons
        l_rate: learning rate
        Batch_size: Batch size
    output:
        score: cross validation MSE
    '''
    N_hidden_nodes = round(N_hidden_nodes)
    Batch_size = round(Batch_size)
    def nn_cl_fun():
        nn = keras.Sequential([
            keras.layers.Dense(N_hidden_nodes, activation=tf.nn.leaky_relu, input_shape=(4,)),
        # keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='linear')
        ])
        nn.compile(loss='mse',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=l_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0),
                    metrics=['mse'])
        return nn
    callbacks = [EarlyStopping(monitor='val_mse', patience=50, verbose=0)]
    nn = KerasRegressor(build_fn=nn_cl_fun, epochs=1000, batch_size=Batch_size,
                        verbose=0)
    # kflod = KFold(n_splits=10, shuffle=True, random_state=123)
    history = nn.fit(Proc_X_train, Proc_Y_train, batch_size=Batch_size, epochs=1000, 
                        verbose=0, validation_split=0.1, callbacks= callbacks)
    score = -history.history['val_loss'][-1]
    return score

def build_model(N_hidden_nodes, input_dim, N_outputs, l_rate, Batch_size, Epochs):
    '''
    Build a NN
    inputs:
        N_hidden_nodes: number of neurons
        input_dim: number of inputs
        N_outputs: number of outputs
        l_rate: learning rate
        Batch_size: Batch size
    outputs:
        model: Trained model
        history: training history
    '''
    callbacks = [
        EarlyStopping(monitor='val_mse', patience=50, verbose=2),
        ModelCheckpoint('best_model.h5', monitor='val_mse', save_best_only=True, verbose=0)
    ]
    model = keras.Sequential([
        keras.layers.Dense(N_hidden_nodes, activation=tf.nn.leaky_relu, input_shape=(input_dim,)),
    # keras.layers.Dropout(0.5),
        keras.layers.Dense(N_outputs,activation='linear')
    ])
    model.compile(loss='mse',
                optimizer=tf.keras.optimizers.Adam(learning_rate=l_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0),
                metrics=['mse'])
    model.summary()
    history = model.fit(Proc_X_train, Proc_Y_train, batch_size=Batch_size, epochs=Epochs, 
                        verbose=1, validation_split=0.1, callbacks= callbacks)
    return model, history

def uniformly_spaced_sampling(y_label, y_label_new, y):
    '''
    fit and interpolate
    '''
    f = interpolate.interp1d(y_label, y, kind='quadratic')
    ynew=pd.DataFrame(f(y_label_new))
    return ynew

def data_melt(x, y):
    '''
    Add position as input
    input:
        x: old input 
        y: old output
    output:
        x_new: new input 
        y_new: new output
    '''
    y_label_str = [str(x) for x in y_label_new]
    dataset = pd.concat([x, y],axis=1, ignore_index=True)
    col_names = ['Travel_length', 'Welding_speed', 'Net_energy_input'] + y_label_str
    dataset.columns = col_names
    dataset = dataset.melt(id_vars=['Travel_length', 'Welding_speed', 'Net_energy_input'], 
        var_name="Position", 
        value_name="Stress")
    x_new = dataset.iloc[:, 0:4]
    y_new = dataset.iloc[:, 4]
    return x_new, y_new

def MSE(testY, predicY):
    '''
    Get MSE fun
    '''
    MSE=np.sum(np.power((testY - predicY),2))/testY.shape[1]/testY.shape[0]
    return MSE

def MAE(testY, predicY):
    '''
    Get MAE fun
    '''
    MAE=np.sum(abs(testY - predicY))/len(testY)
    return MAE

def get_N_output(Y_train):
    '''
    Get the number of outputs
    '''
    if Y_train.ndim == 1:
        N_outputs = 1
    else:
        N_outputs = Y_train.shape[1]
    return N_outputs

def get_y_lable(root):
    '''
    Create an array for uniform interval depth 
    (Used biased mesh in simulaiton, so the sample point depth is not uniform) 
    input:
        root: file root contains the sample point depth information
    outputs:
        y_lab: old sample point position array
        y_lab_new: new sample point position array
    '''
    x, y = data_import(root)
    y_lab = y.iloc[0, :]
    y_lab_new = np.linspace(round(min(y_lab),2), 
                              round(max(y_lab),2), 
                              round(len(y_lab),2))
    return y_lab, y_lab_new
    
def pre_processing(model_type, x, y):
    '''
    Data preprocessing (Uniformly spaced sampling, normalisation, train test split)
    inputs:
        model_type: 'ANN1' the first architecture (3 inputs and 21 outputs)
                    'ANN2' the second architecture (4 inputs and 1 output)        
        x: a dataframe of inputs of the whole dataset
        y: a dataframe of outputs of the whole dataset
    outputs:
        model_type: 'ANN1' the first architecture (3 inputs and 21 outputs)
                    'ANN2' the second architecture (4 inputs and 1 output)
        Proc_X_train: Processed Training input
        Proc_Y_train: Processed Training output
        Proc_X_test: Processed Test input
        Y_test: Test output
    '''
      # uniformly_spaced_sampling
    global y_label, y_label_new
    y_label, y_label_new = get_y_lable(r'extracted_data/benchmark_BD1.csv')
    y = uniformly_spaced_sampling(y_label, y_label_new, y)

    # Split dataset
    # X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=1/125.0, random_state=3)
    X_train, Y_train = data_import(r'extracted_data/S11_along_BD_125.csv')
    Y_train = uniformly_spaced_sampling(y_label, y_label_new, Y_train)
    X_test, Y_test = data_import(r'extracted_data/S11_along_BD_80.csv')
    Y_test = uniformly_spaced_sampling(y_label, y_label_new, Y_test)
    # data reconstruction
    if 'ANN2' in model_type:
        X_train, Y_train = data_melt(X_train, Y_train)
        X_test, Y_test_1 = data_melt(X_test, Y_test)

    # Normalization
    global scaler_X, scaler_Y
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    scaled_train_X = scaler_X.fit_transform(X_train.to_numpy())
    scaled_test_X = scaler_X.transform(X_test)

    if Y_train.ndim == 1:
      Y_train = np.array(Y_train).reshape(-1,1)
    scaled_train_Y = scaler_Y.fit_transform(Y_train)
    Proc_X_train = scaled_train_X
    Proc_Y_train = scaled_train_Y
    Proc_X_test = scaled_test_X

    return model_type, Proc_X_train, Proc_Y_train, Proc_X_test, Y_test

def get_result(model_type, Proc_X_train, Proc_Y_train, Proc_X_test, N_neurons, Batch_size, Epochs, l_rate):
    '''
    Get the trained model, prediction results, and NN training history
    inputs:
        model_type: 'ANN1' the first architecture (3 inputs and 21 outputs)
                    'ANN2' the second architecture (4 inputs and 1 output)
        Proc_X_train: Processed Training input
        Proc_Y_train: Processed Training output
        Proc_X_test: Processed Test input
        N_neurons: Number of Neurons in the hidden layer
        l_rate: Learning rate
    outputs:
        model: Trained model
        predict_test: predicted results
        history: training history
    '''
    #build model
    N_inputs = Proc_X_train.shape[1]
    N_outputs = get_N_output(Proc_Y_train)
    model, history = build_model(N_neurons, N_inputs, N_outputs, l_rate, Batch_size, Epochs)
    # Predict
    predict_test = model.predict(Proc_X_test)
    if 'ANN2' in model_type:
      predict_test = np.reshape(predict_test, (21, -1)).T
    predict_test = scaler_Y.inverse_transform(predict_test)
    # predictdata1 = model.predict(Proc_X_train)
    # predict_data1 = scaler_Y.inverse_transform(predictdata1)
    # plot_error_dist(Proc_X_train, Proc_Y_train, predict_data1)
    # plot_history(history)
    return  model, predict_test, history

def seed_tensorflow(seed):
    '''
    Fix ramdom seed
    '''
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)

# Fix ramdom seed
seed_tensorflow(42)

# Import dataset
x, y = data_import(r'extracted_data/S11_along_BD_205.csv')

# Pre-processing
model_type, Proc_X_train, Proc_Y_train, Proc_X_test, Y_test = pre_processing('ANN2', x, y)

# Run Bayesian Optimization
# param_grid = [{"alpha": [1e-2, 1e-3],
#               "kernel": [RBF(l) for l in np.logspace(-1, 1, 2)]},
#               {"alpha": [1e-2, 1e-3],
#                "kernel":[DotProduct(sigma_0) for sigma_0 in np.logspace(-1, 1, 2)]}
#               ]

# gp = GaussianProcessRegressor(n_restarts_optimizer=10)
# clf = GridSearchCV(estimator=gp, param_grid=param_grid,cv=10,
#                     scoring='mean_squared_error', n_jobs=8)
# clf.fit(Proc_X_train, Proc_Y_train)
# print(clf.best_params_)
# # print(nn_cl_bo(723, 0.000111, 27))
# nn_bo = BayesianOptimization(nn_cl_bo, params_nn, random_state=5)
# nn_bo.maximize(init_points=200, n_iter=20)
# params_nn_ = nn_bo.max['params']
# print(nn_bo)

# Build model and Predict
# ANN, predict_test, history = get_result(model_type, Proc_X_train, Proc_Y_train, Proc_X_test,
#                                       307, 42, 1000, 0.004398)
# MSE_R = MSE(Y_test.values, predict_test)
# print(MSE_R)
# plot_history(history)

# kernel = R(length_scale=1.0, alpha=1.5)
kernel = 1.0 * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e3))
reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, alpha=0.01)
reg.fit(Proc_X_train, Proc_Y_train)


# params_nn ={
#     'N_hidden_nodes': (10, 1000),
#     'l_rate':(0.0001, 0.1),
#     'Batch_size':(1, 64),
# }

# nn_bo = BayesianOptimization(nn_cl_bo, params_nn, random_state=5)
# nn_bo.maximize(init_points=20, n_iter=5)
# params_nn_ = nn_bo.max['params']
# print(nn_bo)

for j in range(80):
    output_data = []
    up_data = []
    down_data = []
    for i in range(21):
        output, err = reg.predict(np.c_[Proc_X_test[j,0].ravel(), Proc_X_test[j,1].ravel(),Proc_X_test[j,2].ravel(),Proc_X_test[j+80*i,3]], return_std=True)
        # output, err = output.reshape(xset.shape), err.reshape(xset.shape)
        # sigma = np.sum(reg.predict(Proc_X_train, return_std=True)[1])
        up, down = output * (1 + 1.96 * err), output * (1 - 1.96 * err)
        output = scaler_Y.inverse_transform(output)
        up = scaler_Y.inverse_transform(up)
        down = scaler_Y.inverse_transform(down)
        output_data.append(output[0][0])
        up_data.append(up[0][0])
        down_data.append(down[0][0])
    # get_plot(Y_test.iloc[0,:], output_data,0)
    plt.figure(figsize=(10, 6))
    plt.plot(y_label_new, output_data, label="GP mean")
    plt.plot(y_label_new, Y_test.iloc[j,:], '--', label="Target")
    plt.fill_between(y_label_new,
                    up_data,
                    down_data,
                    label="95% confidence interval",
                    interpolate=True,
                    facecolor='blue',
                    alpha=0.5)
    plt.legend()
    root = 'H:/PhD_git/ANN_results/BD/test/test--' + str(j) + '.jpg'
    plt.savefig(root)

