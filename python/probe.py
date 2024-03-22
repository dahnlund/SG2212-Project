#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

#mat = loadmat('diag_data.mat')
mat = loadmat("/Users/david/KTH/SG2212-Project/diag_data.mat")
t = mat['t'][0]
u_vec = mat['uvel']
cases = [25, 250, 5000]

du_vec = np.diff(u_vec, axis = 0)/np.diff(t)[0]

plt.figure()
plt.plot(t, u_vec)
plt.plot(t, np.median(u_vec,axis = 0)*np.ones((t.shape[0],3)),'--', color = 'black')
plt.ylabel("U")
plt.xlabel("time")
plt.legend(cases)
plt.grid()
plt.savefig("./plots/plot1.png", bbox_inches='tight')


rel_derivative = du_vec/np.abs(np.median(u_vec,axis = 0))
for col in range(u_vec.shape[1]):
    cond1 = np.abs(u_vec[1:,col]-np.median(u_vec,axis = 0)[col]) < \
        0.05 * np.abs(np.median(u_vec,axis = 0)[col])
    cond2 = np.abs(rel_derivative[:,col]) < 0.05
    print(f'Time to steady flow when Re = {cases[col]}: {t[np.argmax(cond1*cond2)]}')
