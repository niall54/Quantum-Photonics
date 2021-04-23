import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt

class LLE_Solver:
    c = 3e8 # Speed of light
    def __init__(self,
                 a = 1.29e-3,
                 lam0 = 1550e-9,
                 n0 = 1.444,
                 Qext = 1.85e8/2,
                 Q0 = 1.85e8,
                 N = 1024,
                 htau = 0.0002,
                 alpha = 3,
                 beta = -0.005,
                 Pin = 2.7,
                 psi0 = None):
        """
        This object gives a solver for the Lugiato-Lefever equation (LLE). It
        takes "real-world" values for resonator parameters and converts them
        to normalised/dimensionless parameters to be used in the code.
        
        The work is derived from  Yanne Chembo's "Quantum Dynamics of Kerr
        Optical Frequency Combs below and above Threshold: Spontaneous
        Four-Wave-Mixing, Entanglement and Squeezed States of Light" found at
        https://arxiv.org/abs/1412.5700, which explains the normalisation 
        parameters.
        
        Parameters
        ----------
        a : numeric, optional
            Resonator radius (m). The default is 1.29e-3.
        lam0 : numeric, optional
            Input wavelength (m). The default is 1550e-9.
        n0 : numeric, optional
            Refractive index of pump. The default is 1.444.
        Q0 : numeric, optional
            Loaded Q-factor of resonator (half of intrinsic Q-factor for 
            critical coupling). The default is 1.85e8.
        N : int, optional
            Number of simulated modes, must be a power of 2. The default is 
            1024.
        htau : numeric, optional
            Dimensionless simulation time-step, as a proportion of the cavity
            lifetime. The default is 0.0002.
        alpha : numeric, optional
            Dimensionless laser-cavity detuning, normalised by the cavity 
            linewidth. The default is 3.
        beta : numeric, optional
            Dimensionless dispersion parameter. Can be thought of as the shift
            in resonance frequency between adjactent modes (compared to the
            fsr) normalised by the cavity linewidth. The default is -0.005.
        Pin : numeric, optional
            Dimensionless input power, normalised by the Kerr-comb generation
            threshold power. The default is 2.7.

        Returns
        -------
        None.

        """
        #=====================================================================
        #=====================================================================
        # Initialise resonator parameters
        #=====================================================================
        # Resonator input Parameters 
        self.a = a # Radius, m
        self.lam0 = lam0 # Pump wavelength vacuum, m
        self.n0 = n0 # Refraction index @ pump frequency
        self.Q0 = Q0 # Loaded Q-factor
        self.beta = beta # Dimensionless dispersion
        #=====================================================================
        # Pump input parameters
        self.alpha = alpha # Dimensionless cavity detuning
        self.Pin = Pin # Dimensionless input power
        #=====================================================================
        # Resonator calculated Parameters         
        self.w0 = 2*np.pi*self.c/self.lam0 # Pump angular frequency
        self.w_FSR = self.c/(self.a*self.n0) # FSR (angular)
        self.f_FSR = self.w_FSR/(2*np.pi) # FSR (frequency)
        self.dw0 = self.w0/self.Q0 # Linewidth
        self.tau_ph = 1/self.dw0 # Photon lifetime
        #=====================================================================
        #=====================================================================
        # Initialise simulation parameters
        #=====================================================================
        # Simulation input parameters
        self.N = N # Number of simulated modes (must be power of 2)
        assert np.log2(self.N).is_integer(), "Simularion needs 2^n modes"
        self.htau = htau # Dimensionless timestep
        #=====================================================================
        # Simulation calculated parameters
        self.iter = 0 # Iteration count
        self.dtheta = 2*np.pi/(self.N) # Angle step, rad
        self.theta = np.linspace(-np.pi,np.pi,self.N) # Set of angles
        self.ell = np.arange(-self.N/2,self.N/2) # Set of eigennumbers
        self.ell2 = self.ell**2 # Squared eigennumbers
        self.dT = self.dtheta/self.w_FSR # Timestep
        self.Fs = 1/self.dT # Sampling frequency
        self.f = self.ell*self.Fs/self.N # Set of frequencies
        self.tau = 0.0 # Simulation dimensionless time
        self.fExt = self.Pin**0.5 # Input field
        self.fExt_CP = self.fExt # CP input field
        if psi0 is None:
            self.psi0 = 0.8*np.array([1]*self.N) # Initialise the cavity field
        else:
            self.psi0 = psi0
        self.psi = self.psi0
        
    def updateParameter(self,**params):
        """
        This method allows the user to change properties of this object

        Parameters
        ----------
        **params : kwargs
            The keyword arguments will be used to change their respective 
            values. If the keyword does not correspond to a current object 
            property, nothing will be changed and a warning will be displayed.

        Returns
        -------
        None.

        """
        for key in params.keys():
            if key in dir(self):
                self.__dict__[key] = params[key]
            else:
                print('Parameter: "{}" is not defined so this'
                      ' update has been ignored'.format(key))
    
    def runSimulation(self,
                      tauMax = 1000,
                      addRand = True,
                      updateParams = None,
                      saveData = True,
                      saveRate = 100):
        """
        This method runs a simulation using the current object properties. A
        sweep - e.g. of the input frequency to simulate a frequency sweep 
        across a resonance - can be given for any object property using the
        updateParams parameter.

        Parameters
        ----------
        tauMax : numeric, optional
            Length of the simulation time, in terms of the cavity lifetime.
            The default is 1000.
        addRand : bool, optional
            Adds noise to the system to stimulate comb generation when true.
            The default is True.
        updateParams : dict, optional
            A dictionary with all object properties to be swept through during
            the simulation e.g. updateParams = {'Pin':[0,1]} will sweep the 
            input power linearly from 0 at the start of the simulation to 1 at
            the end. The default is None.
        saveData: bool, optional
            When true, the LLE_Solver objects will be saved throughout the 
            simulation at a rate defined by the saveRate parameter. The 
            default is True.
        saveRate: int, optional
            This gives the rate at which the data is saved, in simulation 
            time-steps

        Returns
        -------
        None.

        """
        # ====================================================================
        # Simulation Parameters
        N_sim = int(tauMax/self.htau) # Number of simulation steps
        nProgress = 60 # Progress bar iterations
        progressCount = 0
        saveIndex = 0
        savedIndex = 0
        if saveData:
            numSaves = len(str(int(N_sim/saveRate))) # number of digits 
            timeInfo = datetime.datetime.now()
            prefix = 'data/LLE/'
            prefix += (timeInfo.strftime('%Y%m%d') + '_' 
            +  timeInfo.strftime('%H%M') + '_') 
            
            
        # ====================================================================
        # Generate "sweep" object. This allows for parameters to be 
        # continuously changed throughout the simulation i.e. a frequency 
        # sweep
        updateGenerators = {}
        if updateParams is not None:
            print('Running a simultion in which the following parameters'
                  ' will be swept linearly with time:')
            for param in updateParams.keys(): 
                if param in dir(self):
                    print('{} between {} and {}\n'.format(param, 
                                                        updateParams[param][0],
                                                        updateParams[param][1])
                          )
                    updateGenerators[param] = sweepIterator(
                                                        updateParams[param][0],
                                                        updateParams[param][1],
                                                        N_sim)
                else:
                    print('Could not run sweep for: {}\n'.format(param))
        # ====================================================================  
        print('\n')          
        while self.tau<tauMax:
            # ================================================================
            # Update sweep parameters
            updates = {}
            for param in updateGenerators.keys():
                updates[param] = next(updateGenerators[param])
            self.updateParameter(**updates)
            # ================================================================
            # Run simulation =================================================
            #=================================================================
            # Update linear operator fields
            self.Lin_Op = np.exp((-(1 + 0.0+1.0j*self.alpha) + 
                                  0.0+1.0j*self.beta*self.ell2/2)*self.htau)
            #=================================================================
            # Update cavity fields
            psi_f = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(
                    self.psi)))*self.Lin_Op
            self.psi_f = psi_f
            psi = np.fft.fftshift(np.fft.fft(np.fft.fftshift(psi_f)))
            self.psi = ((psi+self.htau*self.fExt)*
                        (np.exp((1.0j*self.htau*np.abs(psi)**2))))
            if addRand:
                    self.psi += np.random.normal(0,1e-5,self.N)
            # ================================================================
            # Update simulation time 
            self.tau += self.htau
            i = int(self.tau/self.htau)
            if i%int(N_sim/nProgress) ==0:
                progressCount += 1
                print('\rRunning simulation: |{}| {}%'.format(
                    int(progressCount)*'░'+
                    (nProgress-progressCount)*'-',
                    int(100*progressCount/nProgress)),end="")
            # ================================================================
            # Save object (if required)
            saveIndex += 1
            if  (saveIndex % saveRate == 0) and saveData:
                self.save_self(filename = prefix+'{}.pkl'.format(
                    str(savedIndex).zfill(numSaves)))
                savedIndex += 1
                
            
    def plot_self(self,axs=None):
        if axs is None:
            fig = plt.figure()
            fig.subplots_adjust(hspace=0.4)
            ax_cavity  = fig.add_subplot(211)
            ax_modes  = fig.add_subplot(212)
            axs = [ax_cavity, ax_modes]
        
        ax_cavity, ax_modes = axs
        
        ax_cavity.plot(self.theta,
                       np.abs(self.psi)**2,
                       'k')
        ax_cavity.set_xlabel('Longitudinal resonator position/rad')
        ax_cavity.set_ylabel('Intracavity intensity')
        
        ax_modes.plot(self.ell,
                      np.log(np.abs(self.psi_f)**2),
                      'k')
        ax_modes.set_xlabel('Mode number')
        ax_modes.set_ylabel('Modal intensity, dB')
        
    def save_self(self, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
            
class sweepIterator:
    def __init__(self,val0, val1, N):
        """
        This object gives an iterator which will go between two values in N 
        steps.

        Parameters
        ----------
        val0 : numeric
            First value.
        val1 : numeric
            Last value.
        N : numeric (should really be an int)
            Number of steps between the values.

        Returns
        -------
        None.

        """
        self.val = val0
        self.iterVal = (val1-val0)/N
        
    def __iter__(self):
        """
        Initialises the system

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self
    def __next__(self):
        """
        Gives the next value of the iteration when next(self) is called

        Returns
        -------
        x : numeric
            Next value of iterator.

        """
        x = self.val
        self.val += self.iterVal
        return x
    
def load_previous(filename):
    """
    This function simply loads a previously saved version of an LLE_Solver

    Parameters
    ----------
    filename : str
        Filename of a previously saved LLE_Solver object.

    Returns
    -------
    x : LLE_Solver object.

    """
    with open(filename,'rb') as input_:
        x = pickle.load(input_)
    return x
        
if __name__ == '__main__':
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 300
    x = load_previous('data/LLE/SingleSoliton.pkl')
    x.plot_self()