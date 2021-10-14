import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

csv_file_name = 'H:\PhD_data\CSF_results\extracted_data\simualtion_with_error.csv'
raw_data = pd.read_csv(csv_file_name)


data = raw_data.iloc[:,3:6]
x = data.iloc[:,0]
y = data.iloc[:,1]
z = data.iloc[:,2]
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
plt.savefig('./error_analysis.png')
plt.show()
# y.index=x
# plt.plot(y)
# plt.show()