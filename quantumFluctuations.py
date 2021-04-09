import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_lyapunov
from LLE_Solver import *

class QuantumFlux:
    def __init__(self,
                 input_filename = 'SingleSoliton.pkl',
                 input_dir = 'data/LLE/',
                 output_dir = 'data/Quantum Flux/'):
        """
        This object gives a solver for the entanglement matrix for the modes
        of a dissipative Kerr soliton, as described in "Multi-color continuous
        -variable quantum entanglement in dissipative Kerr solitons", Li (2021)
        [This paper will be hereafter referred to as Ref_A]
        
        This requires a previously calculated LLE solution - from the 
        LLE_Solver class in LLE_Solver.py - which has been saved to a .pkl 
        file. This filename (and directory) are the only required inputs.
        
        Use: call x = QuantumFlux(filename, filedir), then x.calculateParams()
        or any of the sub-methods of x.calculateParams() as required

        Parameters
        ----------
        input_filename : TYPE, optional
            Filename for the saved LLE_Solver object. The default is
            'SingleSoliton.pkl'.
        input_dir : TYPE, optional
            Directory of input_dir. The default is 'data/LLE/'.
        output_dir : TYPE, optional
            Directory for saving this object to. The default is 
            'data/Quantum Flux/'.

        Returns
        -------
        QuantumFlux object.
        
        **********************************************************************
        To-Do ****************************************************************
        =====
        - calculate g0 from the imported LLE_Solver object (currently =1)
        - Confirm indexing inside self.makeCouplingMatrix()
        - Confirm elements of D matrix in self.makeDiffMatrix()
        - Confirm elements of M matrix in self.makeCouplingMatrix()
        - Confirm sign of D matrix in self.makeCorrelationMatrix()
        - Confirm indexing inside self.makeLogarithmicNegativityMatrix()
        **********************************************************************
        """
        # Load associated LLE solver object
        self.input_filename = input_filename
        self.input_dir = input_dir
        self.LLE_Soln = load_previous(self.input_dir+self.input_filename)
        self.g0 = 1 # Need to update this!
        # Set directory to save object info to
        self.output_dir = output_dir
        
    def calculateParams(self):
        """
        This method runs through the solver in accordence with "Multi-color
        continuous-variable quantum entanglement in dissipative Kerr solitons",
        Li (2021). Each step is made up of a sub-method for a) ease of reading
        b) allowing the user to call individual 
        Returns
        -------
        Updated QuantumFlux object, with all parameters calculated.

        """
        self.N_lle = int(self.LLE_Soln.N/2) # Number of LLE modes calculated
        self.N = 4*self.N_lle # Number of quadratures to be calculated
        # NB) self.N = 4*self.N_lle because there are two lots of each mode 
        # (one above, one below the pump), each with two quadratures
        
        self.makeBetas() # Make the effective detuning vector, Beta
        self.makeDiffMatrix() # Make the diffusion matrix, D
        self.makeCouplingMatrix() # Make the matrix, M
        self.makeCorrelationMatrix() # Make the correlation matrix, V
        self.makeLogarithmicNegativityMatrix() # Make entanglement matrix, E
        
    def makeBetas(self):
        """
        This (sub-)method makes the Beta vector of Eq. 3 given in the 
        paragraph just following the equation in Ref_A

        Returns
        -------
        Updated QuantumFlux object, with Beta parameter calculated..

        """
        deltaVec = self.LLE_Soln.ell2*self.LLE_Soln.beta + self.LLE_Soln.alpha
        self.Beta = -1.0j*deltaVec + self.LLE_Soln.dw0
        
    def makeDiffMatrix(self):
        """
        This (sub-)method makes the D matrix of Eq. 4 given in the 
        paragraph just following the equation in Ref_A

        Returns
        -------
        Updated QuantumFlux object, with Beta parameter calculated.
        
        **********************************************************************
        To-Do ****************************************************************
        =====
        - Confirm matrix elements. They're currently just set uniformly to the
        linewidth of the cavity (each mode assumed to have the same losses),
        but the exact value may be different to that calculated here
        **********************************************************************

        """
        self.D = np.eye(self.N) * np.sqrt(2*self.LLE_Soln.dw0) # Noise matrix
        
    def makeCouplingMatrix(self):
        """
        This (sub-)method makes the M matrix of Eq. 4 of Ref_A, with a method
        derived from Eq. 3 of Ref_A. This method is given in "Continuous 
        Variable Entanglement.ipynb" under Quantum Fluctuations.

        Returns
        -------
        Updated QuantumFlux object, with the M matrix parameter calculated.
        
        **********************************************************************
        To-Do ****************************************************************
        =====
        - Confirm method derived from Eq. 3, Ref_A. Very fiddly and likely to
        have either errors in the method itself, or typos in the subsequent 
        code
        - Check indexing. I'm currently using a (b) as the row (column) 
        numbers, which each refer to the mode number i~a/2 (j~b/2) as there are
        two rows (columns) for each mode due to there being two quardratures
        for each. The int(a/2)-1 etc. might not be correct! Also check that we 
        correctly run through the correct set of indices - might try this with
        small matrices manually...!
        **********************************************************************

        """
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
                
        # Save this object after a long calculation to avoid needless 
        # repetition
        self.save_self()
    
    def makeCorrelationMatrix(self):
        """
        This (sub-)method solves Eq. 4 of Ref_A for the V matrix. This is an 
        example of the Lyapunov equation so it uses the scipy.linalg method 
        for the solution.

        Returns
        -------
        Updated QuantumFlux object, with the V matrix parameter calculated.
        
        **********************************************************************
        To-Do ****************************************************************
        =====
        - Confirm the sign of the D matrix for the scipy Lyapunov solver.
        **********************************************************************

        """
        self.V = solve_lyapunov(self.M, self.D)
        self.save_self()
        
    def makeLogarithmicNegativityMatrix(self):
        """
        This (sub-)method solves Eq. 5 of Ref_A for the E matrix. 

        Returns
        -------
        Updated QuantumFlux object, with the E matrix parameter calculated.
        
        **********************************************************************
        To-Do ****************************************************************
        =====
        - Confirm the indexing for the A, B, C (C_t) matrices
        **********************************************************************

        """
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
        self.save_self()
        
    def save_self(self, filename=None):
        """
        This method saves the object as a .pkl file to save progress from 
        calculations.

        Parameters
        ----------
        filename : TYPE, optional
            Filename for save. The default is None, which saves the file to
            the output directory and input filename specified when the object
            was initialised.

        Returns
        -------
        None.

        """
        if filename is None:
            filename = self.output_dir + self.input_filename
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