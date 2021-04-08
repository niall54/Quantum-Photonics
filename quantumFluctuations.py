import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_lyapunov
from LLE_Solver import *

class QuantumFlux:
    
    def __init__(self,
                 input_filename = 'SingleSoliton.pkl',
                 input_dir = 'data/LLE/'):
        
        # Load associated LLE solver object
        self.input_filename = input_filename
        self.input_dir = input_dir
        self.LLE_Soln = load_previous(self.input_dir+self.input_filename)
        self.g0 = 1 # Need to update this!
        # Calculate the quantum fluctuation parameters
        self.calculateParams()
        
    def calculateParams(self):
        self.N_lle = int(self.LLE_Soln.N/2) # Number of LLE modes calculated
        self.N = 4*self.N_lle # Number of quadratures to be calculated
        self.makeBetas() # Make the effective detuning vector, Beta
        self.makeDiffMatrix() # Make the diffusion matrix, D
        self.makeCouplingMatrix() # Make the matrix, M
        
    def makeBetas(self):
        deltaVec = self.LLE_Soln.ell2*self.LLE_Soln.beta + self.LLE_Soln.alpha
        self.Beta = -1.0j*deltaVec + self.LLE_Soln.dw0
        
    def makeDiffMatrix(self):
        self.D = np.eye(self.N) * np.sqrt(2*self.LLE_Soln.dw0) # Noise matrix
        
    def makeCouplingMatrix(self):
        ######################################################################
        # FIX ME *************************************************************
        # I just use int() for the sumValComplex without taking into account
        # whether the (a+1)/2 index should just be a/2
        ######################################################################
        self.M = np.zeros((self.N,self.N),dtype=float)
        self.alpha = self.LLE_Soln.psi_f
        
        for i in range(self.N):
            print('\rRunning simulation: {:.2f}%'.format(100*i/self.N),
                  end="")
            a = i + 1 # Row number
            for j in range(self.N):
                # Matrix element m_ij in which (i+1) is the row number and 
                # (j+1) is the column number.
                b = j + 1 # Column number
                m_ab = 0
                
                if a%2 == 1: # a is odd
                    if b == a:
                        m_ab = np.real(self.Beta[int((a+1)/2)-1])
                        m_ab = 0
                    elif b == a+1:
                        m_ab = -np.imag(self.Beta[int((a+1)/2)-1])
                        m_ab = 0
                    else:
                        sumValComplex = 0.0 + 0.0j
                        for d in range(0,2*self.N_lle+1):
                            try: 
                                _= (np.transpose(self.alpha[d])*
                                                  np.transpose(
                                                      self.alpha[int(a/2)+
                                                                 int((b+1)/2)
                                                                 -d])
                                                  -2*np.transpose(
                                                      self.alpha[d])*
                                                  self.alpha[int(a/2)-int(
                                                      (b+1)/2)+d])
                                sumValComplex += 1+1.0j
                                
                            except IndexError:
                                _ = True # The a,b,d indices are not a valid 
                                          # set
                        if b%2 == 1: # b is odd
                            m_ab = self.g0*np.real(sumValComplex)
                        else: #b is even
                            m_ab = self.g0*np.imag(sumValComplex)
                            
                else: # a is even
                    if b == a:
                        m_ab = -np.imag(self.Beta[int(a/2)-1])
                        m_ab = 0
                    elif b == a-1:
                        m_ab = np.real(self.Beta[int(a/2)-1])
                        m_ab = 0
                    else:
                        sumValComplex = 0.0 + 0.0j
                        for d in range(0,2*self.N_lle+1):
                            try: 
                                _= (self.alpha[d]*
                                                      self.alpha[int(
                                                          (a+1)/2)+int((b+1)
                                                                       /2)-d]
                                                  -2*np.transpose(
                                                      self.alpha[d])*
                                                  self.alpha[int((a+1)/2)-
                                                                  int((b+1)/2)
                                                                  +d])
                                                                       
                                sumValComplex+=1+1.0j
                            except IndexError:
                                _ = True # The a,b,d indices are not a valid 
                                          # set
                            
                        if b%2 == 1: # b is odd
                            m_ab = self.g0*np.imag(sumValComplex)
                        else: #b is even
                            m_ab = self.g0*np.real(-sumValComplex)
                
                self.M[i][j] = m_ab
                
        self.save_self(filename = 'data/Quantum Flux/'+self.input_filename)
    
    def makeCorrelationMatrix(self):
        print('Solving Lyuapanov Equation')
        self.V = solve_lyapunov(self.M, self.D)
        self.save_self(filename = 'data/Quantum Flux/'+self.input_filename)
        
    def makeLogarithmicNegativityMatrix(self):
        self.E = np.zeros((self.N_lle,self.N_lle),dtype=float)
        for i in range(self.N_lle):
            for j in range(self.N_lle):
                ## Check indexing for the A, B, C submatrices etc!!
                A = self.V[2*i:2*i+2,2*i:2*i+2]
                B = self.V[2*j:2*j+2,2*j:2*j+2]
                C = self.V[2*i:2*i+2,2*j:2*j+2]
                C_t = self.V[2*j:2*j+2,2*i:2*i+2] # should be transpose of C!
                
                theta = (np.linalg.det(A) + np.linalg.det(B) 
                         - 2*np.linalg.det(C))
                V1 = np.concatenate((A,C),axis=1)
                V2 = np.concatenate((C_t,B),axis=1)
                V = np.concatenate((V1,V2),axis=0)
                eta = np.sqrt(theta - np.sqrt(theta**2 - 4*np.linalg.det(V)))
                self.E[i][j] = np.max([0,-np.log(eta*2**0.5)])
                
        print('Done')
        
    def save_self(self, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
            
if __name__ == '__main__':
    filename = 'SingleSoliton_256.pkl'
    # x = QuantumFlux(input_filename = filename)
    x = pickle.load(open('data/Quantum Flux/'+filename,'rb'))
    x.makeLogarithmicNegativityMatrix()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(x.E)