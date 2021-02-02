import sys
import pickle
import matplotlib.pyplot as plt
from LLE_Solver import *

if __name__ == '__main__':
    fname = sys.argv[1]
    print(fname)
    x = load_previous('data/LLE/'+fname)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(x.theta,abs(x.psi)**2)
    ax2.plot(x.ell,np.log(abs(x.psi_f)**2))
    ax1.set_xlabel('Longitudinal position along resonator')
    ax2.set_xlabel('Mode Number')
    ax1.set_ylabel('Intracavity intensity')
    ax2.set_ylabel('Power spectrum')
    plt.show()