import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csv_file_name = 'H:\PhD_data\CSF_results\extracted_data\simulation_data.csv'
raw_data = pd.read_csv(csv_file_name).dropna()
processed_data = []
for row in range(raw_data.shape[0]):
    for column in range((raw_data.shape[1]-6)):
        processed_data.append([raw_data.iloc[row,0],raw_data.iloc[row,1],raw_data.iloc[row,2],
        raw_data.iloc[row,3],raw_data.iloc[row,4],raw_data.iloc[row,5],column,raw_data.iloc[row,column+6]])
df = pd.DataFrame(processed_data, columns=['bead_length', 'welding_speed', 'heat_input','coefficient','bead_width','bead_depth','position','stress'])
fig = pd.plotting.scatter_matrix(df,figsize=(8,8),c ='blue',marker = 'o',diagonal='',alpha = 0.8,range_padding=0.2)  
plt.savefig('./correlationship.jpg')
plt.show()