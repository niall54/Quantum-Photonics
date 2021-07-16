import os
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
from LLE_Solver import *

def plotGraph(axs, theta, Ell, psi, psi_f):
    for ax in axs: ax.lines = []
    ax1, axPhase, ax2 = axs
    ax1.plot(theta,abs(psi)**2,'k')
    axPhase.plot(theta,np.angle(psi),'k')
    
    modePlot = []
    psiPlot = []
    for index, ell in enumerate(Ell):
        modePlot.append(ell)
        modePlot.append(ell)
        modePlot.append(ell)
        psiPlot.append(-200)
        psiPlot.append(np.log(np.abs(psi_f[index])**2))
        psiPlot.append(np.nan)
    ax2.plot(modePlot,psiPlot,'k')
    

if __name__ == '__main__':
    argv = sys.argv[1:]
    fname = argv[0]
    multiple = False
    try:
        multiple = bool(argv[1])
    except IndexError:
        _=True
        
    print(fname)
    if multiple:
        prefix1, prefix2 = fname.split('_')
        prefix = prefix1 + '_' + prefix2
        print(prefix)
        FNAMES = [prefix+'/'+file for file in os.listdir("data/LLE/"+prefix+'/') if file.endswith('.pkl')]
    else:       
        FNAMES = [fname]
    X = [load_previous('data/LLE/'+fname) for fname in FNAMES]

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    axPhase = fig.add_subplot(312)
    ax2 = fig.add_subplot(313)
    ax1.set_xlabel('Longitudinal position along resonator')
    ylim = 1.1*max([max(abs(x.psi)**2) for x in X])
    ax1.set_ylim([0, ylim])
    ax1.text(1.5,0.75*ylim,'Detuning: {:.2f}'.format(X[0].alpha))
    ax2.set_xlabel('Mode Number')
    ax1.set_ylabel('Intracavity\nintensity')
    axPhase.set_ylabel('Relative\nphase')
    axPhase.set_ylim([-np.pi, np.pi])
    ax2.set_ylabel('Power\nspectrum')
    ax2.set_ylim([-25, 1])
    
    fig2 = plt.figure()
    axSweep = fig2.add_subplot(111)
    for i, x in enumerate(X):
        if i%50 ==0:
            plotGraph(axs=[ax1, axPhase, ax2],
                      theta = x.theta,
                      Ell = x.ell,
                      psi = x.psi,
                      psi_f = x.psi_f)
            psi_tot = 0
            for psi in x.psi:
                psi_tot += np.abs(psi)**2
            axSweep.plot(x.alpha,psi_tot,'k.')
    plt.show()