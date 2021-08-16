import os
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_lyapunov
from LLE_Solver import *

class QuantumLLE:
    
    def __init__(self,
                 lle_file = 'data/LLE/single_mode.pkl',
                 out_file  = 'data/Quantum Flux/SingleSoliton.pkl'):
        
        self.LLE_Soln = load_previous(lle_file)
        self.reduce_LLESoln()
        self.dimensionalise_values()
        self.makeFieldMatrices()
        self.makeQuadratureMatrices()
        self.solveCorrelationMatrix()
        self.getEntanglementMatrix()
        
    def reduce_LLESoln(self):
        """
        The (current) LLE soln object has an asymmetric number of modes, i.e.
        for mode number N it has the modes ell=[-2,-1,0,1] with no +2. The 
        simplest way to deal with this is to remove the first row,column of 
        all the important parts.

        Returns
        -------
        Updated self.LLE_Soln object with symmetric modes.

        """
        updatableParms = ['theta','ell','ell2','psi','psi_f'] # List to update
        for param in updatableParms: # Update all objects in list
            self.LLE_Soln.__dict__[param] = self.LLE_Soln.__dict__[param][1:]
            
        self.LLE_Soln.N -= 1
      
    def dimensionalise_values(self,
                              Pin = 1e-5,
                              rho = 0.999):
        """
        
        Returns
        -------
        None.

        """
        self.rho = rho
        # Switch from linewidth to loss rate
        self.kappa_tot = self.LLE_Soln.dw0/2 # Switch from linewidth to kappa
        # Change the detuning/dispersion terms
        self.sigma = -self.LLE_Soln.alpha * self.kappa_tot
        self.zeta2 = -self.LLE_Soln.beta * self.kappa_tot
        # Find the nonlinear coupling g0 from the input power
        hbar = 6.62607015e-34 # reduced planks constant
        c = 299792458 # speed of light in vacuum
        N_in = Pin/(hbar*c/self.LLE_Soln.lam0) # input photon flux/s
        self.g0 = (self.LLE_Soln.fExt**2 *self.kappa_tot**2 /
                   (2*rho*N_in))
        # Find the photon number-normalised field, A
        self.A_f = self.LLE_Soln.psi_f * np.sqrt(self.kappa_tot/self.g0)
        
    def makeFieldMatrices(self):
        """
        

        Returns
        -------
        None.

        """
        N = len(self.A_f) # Number of modes
        # Initialise Coupling matrices
        R, S = np.zeros((N,N),dtype=complex), np.zeros((N,N),dtype=complex) 
        for i in range(N):
            print('\rCalculating coupling matrix, M: {:.2f}%'.format(100*i/N),
                  end="")
            for j in range(N):
                r_ij = 0
                s_ij = 0
                
                if i-j == 0:
                    # Detuning Term
                    r_ij += - (self.kappa_tot -
                               1.0j*(self.sigma -
                                     0.5*self.zeta2*self.LLE_Soln.ell2[i]))
                    
                # Cycle through to get four-wave mixing terms
                for m in range(N):
                    for n in range(N):
                        if m+i-n-j == 0:
                            r_ij += (2.0j*self.g0*
                                     self.A_f[m]*np.conj(self.A_f[n]))
                        if m+n-i-j == 0:
                            s_ij += 1.0j*self.g0*self.A_f[m]*self.A_f[n]
                            
                # Save elements to matrix
                R[i,j] = r_ij
                S[i,j] = s_ij
        self.R = R
        self.S = S
        
    def makeQuadratureMatrices(self):
        """
        

        Returns
        -------
        None.

        """
        
        self.A = (np.kron(np.array([[1,0],[0,0]]), 
                         np.real(np.conj(self.R)+self.S)) + 
                  np.kron(np.array([[0,1],[0,0]]), 
                         np.imag(np.conj(self.R)+self.S)) +
                  np.kron(np.array([[0,0],[1,0]]), 
                         -np.imag(np.conj(self.R)-self.S)) + 
                  np.kron(np.array([[0,0],[0,1]]), 
                         np.real(np.conj(self.R)-self.S)))
        
        self.D = (np.eye(len(self.A[0])) * 
                  np.sqrt(self.kappa_tot*(
                      1+2*np.sqrt(self.rho*(1-self.rho)))))
        
    def solveCorrelationMatrix(self):
        """
        

        Returns
        -------
        None.

        """
        self.V = solve_lyapunov(self.A, -self.D)
        
    def getEntanglementMatrix(self):
        """
        

        Returns
        -------
        None.

        """
        print('\n')
        
        N = len(self.A_f)
        
        self.E = np.empty((N,N))
        self.E[:] = np.nan
        
        for i in range(N):
            print('\rCalculating entanglement matrix, E: {:.2f}%'.format(
                100*i/N), end="")
            for j in range(N):
                # Get the sub-correlation matrices
                A = np.array([[self.V[i,i], self.V[i+N,i]],
                              [self.V[i,i+N], self.V[i+N,i+N]]])
                
                B = np.array([[self.V[j,j], self.V[j+N,j]],
                              [self.V[j,j+N], self.V[j+N,j+N]]])
                
                C = np.array([[self.V[i,j], self.V[i+N,j]],
                              [self.V[i,j+N], self.V[i+N,j+N]]])
                # Put matrices together
                v_ij = (np.kron(np.array([[1,0],[0,0]]), A) + 
                        np.kron(np.array([[0,1],[0,0]]), C) +
                        np.kron(np.array([[0,0],[1,0]]), np.transpose(C)) + 
                        np.kron(np.array([[0,0],[0,1]]), B))
                # Calculate entanglement parameter
                sumV = (np.linalg.det(A) + np.linalg.det(B) - 
                        2*np.linalg.det(C))
                detV = np.linalg.det(v_ij)
                
                eta_ij = 2**-0.5*np.sqrt(sumV - np.sqrt(sumV**2 - 4*detV))
                
                entVal = -np.log(2*eta_ij)
                if entVal>0:
                    self.E[i,j] = entVal
                    
        fig = plt.figure()
        ax3 = fig.add_subplot(111)
        
        E = ax3.imshow(self.E, origin='lower')
        cb = fig.colorbar(E,ax=ax3)
        ax3.set_title('Entanglement Matrix, $E$')
        
                
                
if __name__ == '__main__':
    lle_file = 'data/LLE/single_soliton2.pkl'
    x = QuantumLLE(lle_file=lle_file)