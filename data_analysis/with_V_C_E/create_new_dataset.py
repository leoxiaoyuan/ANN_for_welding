from re import I
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special.orthogonal import jacobi

def select_parameters(V, A, E):
    result_mat = pd.DataFrame(columns=['voltage', 'current', 'thermol_efficiency', 'heat_input'])
    for voltage in V:
        for current in A:
            for efficiency in E:
                heat_input = voltage*current*efficiency
                if round(heat_input, 2) in [500, 1000, 1500, 2000, 2500, 3000]:
                    result_mat = result_mat.append({'voltage' : voltage, 'current' :current, 
                                    'thermol_efficiency':efficiency,'heat_input':heat_input},ignore_index=True)
    return result_mat

def find_V_C_E(result, heat_input):
    V = []
    C = []
    E = []
    for index, heat_input in enumerate(result.iloc[:, 3]):
        if heat_input == heat_input:
            V.append(result.iloc[index, 0])
            C.append(result.iloc[index, 1])
            E.append(result.iloc[index, 2])

    return V, C, E



V = np.arange(20, 40.1, 0.5)
A = np.arange(100, 200.1, 1)
E = np.arange(0.7, 0.9 , 0.01)

result = select_parameters(V, A, E)
csv_file_name = r'extracted_data\simulation_data_with_V_C_E.csv'
raw_data = pd.read_csv(csv_file_name).dropna()

new_data = pd.DataFrame(columns=('bead_length', 'voltage', 'current', 'travel_speed', 
                                'thermol_efficiency','energy_input', 'bead_width', 'bead_depth'))
columns = np.arange(1,62,1)
i = 0
stress_data = pd.DataFrame(columns = columns)
for index in range(len(raw_data)):
    heat_input = raw_data.iloc[index,2]
    travel_speed = raw_data.iloc[index,2]/1000
    V, C, E = find_V_C_E(result,heat_input)
    for index_2 in range(len(V)):
        new_data = new_data.append({'bead_length': raw_data.iloc[index,0],'voltage' : V[index_2], 'current' :C[index_2], 
                            'travel_speed': raw_data.iloc[index,1], 'thermol_efficiency':E[index_2],
                            'energy_input':raw_data.iloc[index,2], 'bead_width':raw_data.iloc[index,4],
                            'bead_depth': raw_data.iloc[index,5]},ignore_index=True)
        stress_data.loc[i] = raw_data.iloc[index,6:].values
        i += 1
new_data = pd.concat([new_data, stress_data],axis=1)
new_data.to_csv(r'extracted_data/simulation_data_with_V_C_E.csv', index=False)
