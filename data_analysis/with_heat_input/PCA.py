import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

csv_file_name = r'extracted_data/simulation_data_with_heat_input.csv'
raw_data = pd.read_csv(csv_file_name).dropna()
processed_data = []
for row in range(raw_data.shape[0]):
    for column in range((raw_data.shape[1]-6)):
        processed_data.append([raw_data.iloc[row,0],raw_data.iloc[row,1],raw_data.iloc[row,2],
        raw_data.iloc[row,3],raw_data.iloc[row,4],raw_data.iloc[row,5],column,raw_data.iloc[row,column+6]])
df = pd.DataFrame(processed_data, columns=['bead_length', 'welding_speed', 'heat_input','coefficient','bead_width','bead_depth','position','stress'])

df1 = df.loc[:,df.columns[0:7]]
scaler = StandardScaler()
scaler.fit(df1)
scaled_df = scaler.transform(df1)
pca = PCA(n_components = 7)
pca.fit(scaled_df)
x_pca = pca.transform(scaled_df)
percent_variance = pca.explained_variance_ratio_
x = np.arange(7)
plt.bar(x,height=percent_variance,width=0.5,)
plt.ylabel('Percentate of Variance Explained')
plt.xlabel('Principal Component')
plt.title('PCA Scree Plot')
plt.show()