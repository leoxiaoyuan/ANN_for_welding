# from cmath import sqrt,exp  
from enum import EnumMeta
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib import cm
import matplotlib.colors as mcolors
bead_length = 0.07
stop_time_at_start = 0
welding_time = 17.5
arc_speed = 0.004
heat_input = 3000
f1=1
f2=2.0-f1
PI=3.1415926
heat_source_size_coeff = 1
a1=0.0045 * heat_source_size_coeff
a2=0.0065 * heat_source_size_coeff
b=0.0068 * heat_source_size_coeff
c=0.0026 * heat_source_size_coeff
result = pd.DataFrame(columns=['x', 'y', 'z', 'flux'])
for x in np.arange(-0.09,0.09,0.0015):
    for y in np.arange(0,0.06,0.0015):
        for z in np.arange(0,0.017,0.0015):
            heat1=6.0*np.sqrt(3.0)*heat_input/(a1*b*c*PI*np.sqrt(PI))*f1
            heat2=6.0*np.sqrt(3.0)*heat_input/(a2*b*c*PI*np.sqrt(PI))*f2
            shape1=np.exp(-3.0*(x-0.03)**2/a1**2-3.0*(y-0)**2/b**2-3.0*(z-0)**2/c**2)
            shape2=np.exp(-3.0*(x-0.03)**2/a2**2-3.0*(y-0)**2/b**2-3.0*(z-0)**2/c**2)
            if x > 0.03:
                flux = heat1 * shape1   
            else:
                flux = heat2 * shape2
            result = result.append({'x' : x, 'y' :y, 'z': z, 'flux':flux},ignore_index=True)

# print(result)
x = result.iloc[:,0]
y = result.iloc[:,1]
z = result.iloc[:,2]
V = result.iloc[:,3]
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z, c = V, cmap="rainbow")
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