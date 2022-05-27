from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd  
raw_data = pd.read_csv('H:/PhD_git/extracted_data/Parameter_sapce/S11_along_BD_205.csv',header=None).dropna()
a = raw_data.iloc[:,0]
b = raw_data.iloc[:,1]
c = raw_data.iloc[:,2]
d = raw_data.iloc[:,3]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# X_train = scaler_X.inverse_transform(X_train)
# Y_train = scaler_Y.inverse_transform(Y_train)
# for i in range(len(X_train)):
#     a.append(X_train[i,0])
#     b.append(X_train[i,1])
#     c.append(X_train[i,2])
#     MSE_R = MSE(Y_train[i,:], predict_data1[i,:])
#     d.append(MSE_R)
img = ax.scatter(a, b, c, c=d, cmap=plt.get_cmap('rainbow'))
# # ax.plot_trisurf(b, c, d)
fig.colorbar(img)
plt.show()