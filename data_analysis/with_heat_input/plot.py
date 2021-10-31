import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

csv_file_name = r'extracted_data/S11_along_BD.csv'
raw_data = pd.read_csv(csv_file_name, header=None).dropna()

# plot stress
x = raw_data.iloc[0,6:]
y = raw_data.iloc[1:,6:].T
y.index=x
plt.plot(y)
plt.show()

# plot bead width and bead depth
x = np.arange(149)
bead_width = raw_data.iloc[:,4].T
bead_depth = raw_data.iloc[:,5].T
bead_width.index=x
bead_depth.index=x
plt.plot(bead_width, label = 'bead_width')
plt.plot(bead_depth, label = 'bead_depth')
plt.legend()
plt.show()