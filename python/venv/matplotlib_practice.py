import random
import numpy as np
import matplotlib.pyplot as plt

t=np.arange(0,5,0.01)
s=np.cos(2*t*np.pi)
plt.plot(t,s,lw=2)
plt.annotate('local maxima', xy=(2,1), xytext=(3,1.5),
             arrowprops=dict(facecolor='blue', shrink=0.05))
plt.ylim(-2,2)
plt.show()