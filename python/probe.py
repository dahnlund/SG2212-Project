import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

mat = loadmat('diag_data.mat')
t = mat['t'][0]
uvel = mat['uvel']
cases = [25, 250, 5000]

duvel = np.diff(uvel, axis = 0)/np.diff(t)[0]

print(np.median(uvel,axis = 0))

plt.figure()
plt.plot(t, uvel)
plt.plot(t, np.median(uvel,axis = 0)*np.ones((t.shape[0],3)),'--', color = 'black')
plt.ylabel("U")
plt.xlabel("time")
plt.legend(cases)
plt.grid()
plt.savefig("./plots/plot1.png", bbox_inches='tight')


rel_deriv = duvel/np.median(uvel,axis = 0)
plt.figure()
plt.plot(t[1:], rel_deriv)
plt.ylabel("U")
plt.xlabel("time")
plt.legend(cases)
plt.grid()
plt.savefig("./plots/plot1derivative.png", bbox_inches='tight')



print(np.abs(np.argmax(np.abs(rel_deriv[:,1]-np.median(uvel,axis = 0)[1])<0.01)))
for col in range(uvel.shape[1]):
    print(t[np.argmax(np.abs(rel_deriv[:,col]) < 0.01)])
