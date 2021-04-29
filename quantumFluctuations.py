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
        input_filename : str, optional
            Filename for the saved LLE_Solver object. The default is
            'SingleSoliton.pkl'.
        input_dir : str, optional
            Directory of input_dir. The default is 'data/LLE/'.
        output_dir : str, optional
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
        self.reduce_LLESoln() # Remove unwanted initial mode
        self.get_g0()
        # Chembo and Quantum Correlation papers
        # Set directory to save object info to
        self.output_dir = output_dir
        
    def get_g0(self,pThresh=1e-3):
        """
        This routine calculates the g0 - Kerr coupling strength - from the 
        LLE parameters and a (given) intended threshold power. This threshold
        power is just to dimensionalise the system such that comb generation
        only happens for large cavity photon numbers (1uW seems typical)

        Parameters
        ----------
        pThresh : numeric, optional
            The threshold power for comb generation in watts. The default is
            1e-3 (1mW).

        Returns
        -------
        Updated QuantumFlux object with g0.

        """
        hbar = 6.62607015e-34 # reduced planks constant
        g0_chembo = self.LLE_Soln.dw0**2*(hbar*self.LLE_Soln.w0)/(16*pThresh)
        self.g0 = -g0_chembo # Difference in definition of g0 for Chembo/Li
        
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
        self.N = 4*self.N_lle + 2 # Number of quadratures to be calculated
        # NB) self.N = 4*self.N_lle because there are two lots of each mode 
        # (one above, one below the pump), each with two quadratures
        
        self.makeBetas() # Make the effective detuning vector, Beta
        self.makeDiffMatrix() # Make the diffusion matrix, D
        self.makeCouplingMatrix() # Make the matrix, M
        self.makeCorrelationMatrix() # Make the correlation matrix, V
        self.makeLogarithmicNegativityMatrix() # Make entanglement matrix, E
        ######################################################################
        # Delete me, just for checking stuff
        self.checkDiagonals()
        ######################################################################
        
    def makeBetas(self):
        """
        This (sub-)method makes the Beta vector of Eq. 3 given in the 
        paragraph just following the equation in Ref_A

        Returns
        -------
        Updated QuantumFlux object, with Beta parameter calculated..

        """
        deltaVec = self.LLE_Soln.ell2*self.LLE_Soln.beta/2 - self.LLE_Soln.alpha
        self.Beta = (-1.0j*deltaVec + 1)*self.LLE_Soln.dw0/2
        
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
        self.D = np.eye(self.N) * self.LLE_Soln.dw0/2 # Noise matrix
        self.D *= self.LLE_Soln.dw0/2 # Not sure why this is here...
        
    def makeCouplingMatrix(self):
        """
        This (sub-)method makes the M matrix of Eq. 4 of Ref_A, with a method
        derived from Eq. 3 of Ref_A. This method is given in "Continuous 
        Variable Entanglement.ipynb" under Quantum Fluctuations.

        Returns
        -------
        Updated QuantumFlux object, with the M matrix parameter calculated.
        **********************************************************************

        """
        self.alpha = (self.LLE_Soln.psi_f)*(self.LLE_Soln.dw0/(2*self.g0))**0.5
        N = len(self.alpha)
        # This next removes the extra strength of the pump mode
        A = np.zeros((N,N))
        B = np.zeros((N,N))
        C = np.zeros((N,N))
        D = np.zeros((N,N))
        for I in range(N):
            print('\rRunning simulation: {:.2f}%'.format(100*I/N),
                  end="")
            for J in range(N):
                if I == J:
                    A[I][J] += np.real(self.Beta[I])
                    B[I][J] -= np.imag(self.Beta[I])
                    C[I][J] += np.imag(self.Beta[I])
                    D[I][J] += np.real(self.Beta[I])
                sum1 = 0
                sum2 = 0
                
                for c in range(N):
                    if I+J-c>=0:
                        try:
                            sum1 += self.alpha[c]*self.alpha[I+J-c]
                        except IndexError:
                            _ = True
                    if c+I-J>=0:
                        try:
                            sum2 += self.alpha[c]*self.alpha[c+I-J]
                        except IndexError:
                            _ = True
                A[I][J] += self.g0*(np.imag(sum1) - 2*np.imag(sum2))
                B[I][J] -= self.g0*(np.real(sum1) - 2*np.real(sum2))
                C[I][J] -= self.g0*(np.real(sum1) + 2*np.real(sum2))
                D[I][J] -= self.g0*(np.imag(sum1) + 2*np.imag(sum2))
                        
        self.A_ = A      
        self.B_ = B      
        self.C_ = C      
        self.D_ = D
                    
        self.M = ( np.kron(A,np.array([[1,0],[0,0]]))
                  +np.kron(B,np.array([[0,1],[0,0]]))
                  +np.kron(C,np.array([[0,0],[1,0]]))
                  +np.kron(D,np.array([[0,0],[0,1]])))
        
        self.M *= self.LLE_Soln.dw0/2 # Swap from chembo's tau to t
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
        self.V = solve_lyapunov(self.M, -self.D)
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
        self.E = np.zeros((self.N_lle*2+1,self.N_lle*2+1),dtype=float)
        detV = np.linalg.det(self.V)
        
        for i in range(self.N_lle*2+1):
            for j in range(self.N_lle*2+1):
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
                eta = np.sqrt(theta - np.sqrt(theta**2 - 
                                              4*np.linalg.det(V)))
                self.E[i][j] = np.max([0,-np.log(eta*2**0.5)])
                if self.E[i][j]<=0:
                    self.E[i][j] = np.nan
        self.save_self()
        
    def save_self(self, filename=None):
        """
        This method saves the object as a .pkl file to save progress from 
        calculations.

        Parameters
        ----------
        filename : str, optional
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
            
    def checkDiagonals(self):
        """
        This just plots what would be expected for the A_, B_, C_ and D_ 
        matrices based off a single pumped field to check for differences 
        between the calculated versions

        Returns
        -------
        None.

        """
        N = len(self.alpha)
        alpha_0 = x.alpha[int(N/2)]
        
        A = np.zeros((N,N))
        B = np.zeros((N,N))
        C = np.zeros((N,N))
        D = np.zeros((N,N))
        
        for i in range(N):
            A[i][i] += np.real(self.Beta[i])-2*self.g0*np.imag(alpha_0*alpha_0)
            A[i][N-1-i] += self.g0*np.imag(alpha_0*alpha_0)
            
            B[i][i] += -np.imag(self.Beta[i])+2*self.g0*np.real(alpha_0*alpha_0)
            B[i][N-1-i] += -self.g0*np.real(alpha_0*alpha_0)
            
            C[i][i] += np.imag(self.Beta[i])-2*self.g0*np.real(alpha_0*alpha_0)
            C[i][N-1-i] += -self.g0*np.real(alpha_0*alpha_0)
            
            D[i][i] += np.real(self.Beta[i])-2*self.g0*np.imag(alpha_0*alpha_0)
            D[i][N-1-i] += -self.g0*np.imag(alpha_0*alpha_0)
            
        letts = [A,B,C,D]
        letts_ = [self.A_, self.B_, self.C_, self.D_]
        
        fig = plt.figure()
        for INDEX, let in enumerate(letts):
            let_ = letts_[INDEX]
            axlet = fig.add_subplot(4,3,1+INDEX*3)
            axlet_ = fig.add_subplot(4,3,2+INDEX*3)
            axDiff = fig.add_subplot(4,3,3+INDEX*3)
            pltlet = axlet.imshow(let)
            pltlet_ = axlet_.imshow(let_)
            pltDiff = axDiff.imshow(let-let_)
            fig.colorbar(pltDiff, ax=axDiff)
            
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        E = ax2.imshow(self.E,origin='lower')
        fig2.colorbar(E,ax=ax2)
        
# def analyzeObject(filename = 'single_mode.pkl'):
#     with open('data/quantum flux/'+filename,'rb') as input_:
#         x = pickle.load(input_)
        
    
#     i = 1
#     j = 101
    
#     A = x.V[2*i:2*i+2,2*i:2*i+2]
#     B = x.V[2*j:2*j+2,2*j:2*j+2]
#     C = x.V[2*i:2*i+2,2*j:2*j+2]
#     C_t = x.V[2*j:2*j+2,2*i:2*i+2]
    
#     for zz in [A,B,C]:
#         print(zz,  np.linalg.det(zz))
#     theta = np.linalg.det(A) + np.linalg.det(B) - 2*np.linalg.det(C)
#     print(theta)
    
if __name__ == '__main__':
    filename = 'single_soliton_efficient.pkl'
    plt.close('all')
    x = QuantumFlux(input_filename=filename)
    x.calculateParams()
    # analyzeObject()