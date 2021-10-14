import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib import cm

##process data and save it to txt
# csv_file_name = ['H:\PhD_data\CSF_results\extracted_data\simulation_data_for_whole_500.csv',
#                  'H:\PhD_data\CSF_results\extracted_data\simulation_data_for_whole_1000.csv',
#                  'H:\PhD_data\CSF_results\extracted_data\simulation_data_for_whole_1500.csv',
#                  'H:\PhD_data\CSF_results\extracted_data\simulation_data_for_whole_2000.csv',
#                  'H:\PhD_data\CSF_results\extracted_data\simulation_data_for_whole_2500.csv',
#                  'H:\PhD_data\CSF_results\extracted_data\simulation_data_for_whole_3000.csv',]
# all_data = pd.DataFrame() 
# for csv_file in csv_file_name:
#     raw_data = pd.read_csv(csv_file, names=['info', 'x', 'y', 'z', 'stress']).dropna()
#     all_data=all_data.append(raw_data,ignore_index=True)

# info_data = all_data['info'].str.split('-', expand=True)
# info_data.columns = ['bead_lenth', 'welding_speed', 'heat_input', 'coefficient']
# all_data = pd.concat([all_data, info_data], axis=1)
# stress_data = all_data['stress']
# all_data.drop(columns = ['info','stress'],inplace = True)
# all_data['stress'] = stress_data
# all_data.to_csv('extracted_data/simulation_data_for_whole.txt', index=False, sep='\t')

# read data from txt
all_data = pd.read_csv('extracted_data/simulation_data_for_whole.txt', sep='\t')
print(all_data.head())

data = all_data.iloc[:14030,:]
x = data.iloc[:,0]
y = data.iloc[:,1]
z = data.iloc[:,2]
V = data.iloc[:,7]
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z, c = V, cmap="rainbow")
# gradient = np.linspace(min(V), max(V), 2)
# gradient = np.vstack((gradient, gradient))
# fig, ax_legend = plt.subplots(nrows=1, figsize=(6.4, 1))
# fig.subplots_adjust(top=1-.35/1, bottom=.15/1, left=0.2, right=0.99)
# ax_legend.imshow(gradient, aspect='auto', cmap='rainbow')
m = cm.ScalarMappable(cmap=cm.rainbow)
m.set_array(V)
plt.colorbar(m, label=" s11",fraction=0.03, ticks=np.linspace(min(V), max(V), 5)) 

# Labels
ax.set_xlabel('X')
ax.set_xlim3d(-0.1, 0.1)
ax.set_ylabel('Y')
ax.set_ylim3d(-0.1, 0.1)
ax.set_zlabel('Z')
ax.set_zlim3d(-0.1, 0.1)

plt.show()