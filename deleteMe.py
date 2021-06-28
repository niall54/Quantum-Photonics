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
    plt.close('all')
    argv = sys.argv[1:]
    
    fig = plt.figure(figsize=(8,10))
    ax1 = fig.add_subplot(3,2,1)
    ax2 = fig.add_subplot(3,2,2)
    ax3  = fig.add_subplot(3,2,3)
    ax4  = fig.add_subplot(3,2,4)
    ax5  = fig.add_subplot(3,1,3)
    axs = [ax1,ax2,ax3,ax4]
    
    directories = ['20210615_1859/',
                   '20210615_1950/',
                   '20210615_2016/',
                   '20210615_2041/']
    
    Ns = ['2500.pkl','3250.pkl']
    
    for i,N in enumerate(Ns):
        fname = 'data/LLE/20210615_2107/'+N
        sim = load_previous(fname)
        sim.tau = 0
        sim.runSimulation(tauMax=1000,addRand=False,saveData=False)
        ax4.plot([sim.alpha,sim.alpha],[0,1],'k--',alpha=0.5)
        ax5.plot(sim.ell,np.log(np.abs(sim.psi_f)**2))
        ax5.set_ylim([-25,5])
        
    for i, directory in enumerate(directories):
        
        fname = directory
        ax =  axs[i]
        FNAMES = [file for file in os.listdir("data/LLE/"+directory)]
        X = ['data/LLE/'+directory+fname for fname in FNAMES]
        alphas = []
        pwrs = []
        for i, x in enumerate(X):
            sim = load_previous(x)
            totalPwr = np.sum(np.abs(sim.psi)**2)/sim.N
            pwrs.append(totalPwr/sim.Pin)
            alphas.append(sim.alpha)
        
        ax.plot(alphas,pwrs,label='$P_\text{in}$='+str(sim.Pin))
        ax.set_xlim([-4,4])
        ax.set_ylim([-0,1.1])
    plt.show()