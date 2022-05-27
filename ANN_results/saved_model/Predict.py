from keras.models import load_model
import pickle
import numpy as np
import pandas as pd

scaler_X = pickle.load(open(r'data_analysis\saved_model\scaler_X.pkl', 'rb'))
scaler_Y = pickle.load(open(r'data_analysis\saved_model\scaler_Y.pkl', 'rb'))

# Define the welding parameters
Input_dist = {
      'Travel length': [60], 
      'Welding speed': [2270],
      'Net energy input': [1100],
      'Position': np.linspace(0, 17, 21)
}

# Load ANN model model and predict
Input = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in Input_dist.items()])).fillna(method='pad')
Proc_X = scaler_X.transform(Input.values.reshape(-1,4)).reshape(-1, 4)
model = load_model(r'data_analysis\saved_model\ANN.h5')
# model.summary()
predict_test_scal = model.predict(Proc_X)
Proc_Y = scaler_Y.inverse_transform(predict_test_scal.reshape(-1,21))[0]
print(Input_dist)