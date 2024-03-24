#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

#mat = loadmat('diag_data.mat')
mat = loadmat("./diag_data.mat")
t = mat['t'][0]
u_vec = mat['uvel']
cases = [25, 250, 5000]

du_vec = np.diff(u_vec, axis = 0)/np.diff(t)[0]
ddu_vec = np.diff(u_vec, axis = 0, n = 2)/(np.diff(t)[0])**2

saved_points = np.zeros((len(cases),2))

for col in range(u_vec.shape[1]):
    cond1 = np.abs(u_vec[2:,col]-np.median(u_vec,axis = 0)[col]) < \
        0.05 * np.abs(np.median(u_vec,axis = 0)[col])
    cond2 = np.abs(du_vec[1:,col]) < 0.001
    cond3 = np.abs(ddu_vec[:,col]) < 0.001

    cond = cond1*cond2*cond3

    index = np.argmax(cond)
    saved_points[col,0] = t[index]
    saved_points[col,1] = u_vec[index, col]

    print(f'Time to steady flow when Re = {cases[col]}: {t[index]}')

plt.figure()
plt.plot(t, u_vec)
plt.plot(t, np.median(u_vec,axis = 0)*np.ones((t.shape[0],3)),'--', color = 'black')
plt.scatter(saved_points[:,0], saved_points[:,1], color = 'red')
plt.ylabel("U")
plt.xlabel("time")
plt.legend(cases)
plt.grid()
plt.savefig("./plots/plot1.png", bbox_inches='tight')
