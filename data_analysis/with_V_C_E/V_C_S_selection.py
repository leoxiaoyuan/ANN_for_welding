from enum import EnumMeta
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib import cm
import matplotlib.colors as mcolors

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

V = np.arange(10, 40.1, 0.5)
A = np.arange(50, 300.1, 1)
E = np.arange(0.7, 0.9 , 0.01)
result = select_parameters(V, A, E)
print(result)
x = result.iloc[:,0]
y = result.iloc[:,1]
z = result.iloc[:,2]
value = result.iloc[:,3]
fig = plt.figure()
ax = Axes3D(fig)
cmap = mcolors.ListedColormap([(0.8941176470588236, 0.10196078431372549, 0.10980392156862745), (0.21568627450980393, 0.49411764705882355, 0.7215686274509804), (0.30196078431372547, 0.6862745098039216, 0.2901960784313726), (0.596078431372549, 0.3058823529411765, 0.6392156862745098), (1.0, 0.4980392156862745, 0.0), (1.0, 1.0, 0.2)])
ax.scatter(x, y, z, c = value, cmap=cmap)
m = cm.ScalarMappable(cmap=cmap)
m.set_array(V)
plt.colorbar(m, label=" s11",fraction=0.03, ticks=np.linspace(min(value), max(value), 5)) 
plt.show()