import os
import cv2
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from LLE_Solver import *
from quantumFluctuations import *

def LLE_Sweep_Plotter(directory = None,
                      axs = None,
                      makeVid = True,
                      plotRate = 10):
    dir_prefix = 'data/LLE/'
    
    if not os.path.isdir(dir_prefix + directory + '/figs'):
        os.mkdir(dir_prefix + directory + '/figs')
        
    if axs == None:
        fig = plt.figure(figsize=(10,10))
        gs = fig.add_gridspec(2,4)
        ax1 = fig.add_subplot(gs[0,0:2])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[1,1])
        ax4 = fig.add_subplot(gs[:,2:])
    
    dets = []
    pwrs = []
    maxPsi_f = 0
    maxPsi = 0
    for file in os.listdir(dir_prefix + directory):
        if file.endswith('.pkl'):
            x = load_previous(dir_prefix + directory +'/'+ file)
            dets.append(x.alpha)
            pwrs.append(np.sum(np.abs(x.psi)))
            maxPsi = max(maxPsi, max(abs(x.psi)**2))
            maxPsi_f = max(maxPsi, max(np.log(abs(x.psi_f)**2)))
        
        
    pngFiles = []
    
    for idx, file in enumerate(os.listdir(dir_prefix + directory)):
        if file.endswith('.pkl'):
            if idx%plotRate == 0:
                ax1.lines = []
                ax2.lines = []
                ax3.lines = []
                ax1.plot(dets,pwrs,'k')
                ax1.plot(dets[idx], pwrs[idx], 'k.')
                x = load_previous(dir_prefix + directory +'/'+ file)
                ax2.plot(x.ell, np.log(np.abs(x.psi_f)**2),'k')
                ax3.plot(np.abs(x.psi)**2,'k')
                ax2.set_ylim([-40,maxPsi_f])
                ax3.set_ylim([0,maxPsi])
                
                y =  QuantumFlux(input_filename= directory +'/'+ file,
                                 output_dir='data/Quantum Flux/')
                y.calculateParams(plot_results=False)
                E = ax4.imshow(y.E, origin='lower',
                               extent=y.N_lle*np.array([-1,1,-1,1]))
                cb = fig.colorbar(E,ax=ax4)
                ax4.set_title('Entanglement Matrix, $E$')
                
                pngFile = dir_prefix + directory + '/figs/' + file.split('.')[0] +'.png'
                fig.savefig(pngFile)
                pngFiles.append(pngFile)
                cb.remove()
    img_array = []        
    for filename in pngFiles:        
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
     
    vidFile = dir_prefix + directory + '/figs/sweep.avi' 
    out = cv2.VideoWriter(vidFile,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
        
if __name__ == '__main__':
    plt.close('all')
    directory = '20210709_1701'
    LLE_Sweep_Plotter(directory=directory,plotRate=100)
    # directory = 'data\\quantum flux\\soliton_sweep_maybe'
    # dir_end = directory.split('\\')[-1]
    
        
    # for file in os.listdir(directory):
    #     plt.close('all')
        
    #     fig = plt.figure(dpi=300)
    #     fig.subplots_adjust(hspace=0.4)
    #     gs = fig.add_gridspec(2,3)
    #     ax1 = fig.add_subplot(gs[0,0])
    #     ax2 = fig.add_subplot(gs[1,0])
    #     ax3 = fig.add_subplot(gs[:,1:])
    #     axs = [ax1, ax2, ax3]
        
    #     fname = directory + '\\' +file
    #     print(fname)
    #     x = load_previous(fname)
    #     x.plotResults(fig = fig, axs=axs)
        
    #     fig.savefig(directory+'/'+file.split('.')[0]+'.png')
    
        