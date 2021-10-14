import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib import cm

# read data from txt
all_data = pd.read_csv('extracted_data/simulation_data_new.txt', sep='\t')
print(all_data.head())

