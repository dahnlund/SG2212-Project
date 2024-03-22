import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

mat = loadmat('diag_data.mat')
t = mat['t'][0]
uvel = mat['uvel']
cases = [25, 250, 5000]

plt.figure()
plt.plot(t, uvel)
plt.ylabel("U")
plt.xlabel("time")
plt.legend(cases)
plt.grid()
plt.savefig("./plots/plot1derivative.png", bbox_inches='tight')