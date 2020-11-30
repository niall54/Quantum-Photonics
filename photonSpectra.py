import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QWidget,
                             QPushButton, QVBoxLayout)
import PyQt5.QtWidgets as QtWidgets
from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=100, dpi=200):
        """
        This object gives a matplotlib canvas to be used for plotting in a 
        PyQt environment

        Parameters
        ----------
        parent : Gui instance, optional
            This is the window in which the canvas will be displayed.
            The default is None.
        width : Numeric, optional
            Width of canvas. The default is 5.
        height : Numeric, optional
            Height of canvas. The default is 100.
        dpi : Numeric, optional
            Resolution of canvas. The default is 200.

        Returns
        -------
        None.

        """
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        
class MyLabel(QWidget):
    def __init__(self, name = 'test', value = 1, varName = 'test',
                 parent=None):
        """
        This object gives an editable label for PyQt.

        Parameters
        ----------
        name : String, optional
            This is the string that will describe the label in the window.
            The default is 'test'.
        value : Numeric, optional
            This is the value of the parameter. The default is 1.
        varName : String, optional
            This is the variable name associated with the label.
            The default is 'test'.
        parent : Gui instance, optional
            This is the object who's attribute "varName" is set by this label.
            The default is None.

        Returns
        -------
        None.

        """
        
        super(MyLabel,self).__init__()
        self.hbox = QtWidgets.QHBoxLayout()
        self.hbox.addWidget(QtWidgets.QLabel(name+' :'))
        self.value = QtWidgets.QLineEdit(str(value))
        self.hbox.addWidget(self.value)
        self.setLayout(self.hbox)
        
        self.value.returnPressed.connect(self.valueUpdated)
        
        self.varName = varName
        self.parent = parent
        
    def valueUpdated(self):
        """
        This method changes the variable in the parent object to the new value

        Returns
        -------
        None.

        """
        try:
            self.parent.updateOptionValue(self.varName,
                                          float(self.value.text()))
        except:
            print('Could not change parameter')
        
class MyDisplayWidget(QWidget):
    def __init__(self, nIn = 1, nOut = 2):
        """
        This object gives a display widget showing the photon flux into the 
        resonator and out of the chosen mode. The relative isolation of the 
        pump to equal the fluxes is also displayed, obviously this will give
        a signal-to-noise ratio of 1, which is poor, but gives an idea of the
        required isolation.
        
        NB) This assumes that all input photons will go to the photo-detector,
        though this is obviously incorrect (in the case of critical coupling,
        none of them will and no pump suppression is required) but is a worst
        case scenario. 

        Parameters
        ----------
        nIn : Numeric, optional
            Number of input photons. The default is 1.
        nOut : Numeric, optional
            Number of photons output from mode. The default is 2.

        Returns
        -------
        None.

        """
        super(MyDisplayWidget,self).__init__()
        self.hbox = QtWidgets.QHBoxLayout()
        reqDb = 10*np.log10(nIn/nOut)
        
        str1 = 'Photons in: {}\nPhotons out: {}'.format(int(nIn),int(nOut))
        str2 = '\n\nRequired isolation: {}dB'.format(int(reqDb))
        self.Label = QtWidgets.QLabel(str1+str2)
        self.hbox.addWidget(self.Label)
        self.setLayout(self.hbox)
        
    def updateDisplay(self,nIn, nOut):
        """
        This method updates the display, upon changes to nIn, nOut

        Parameters
        ----------
        nIn : Numeric
            Number of input photons.
        nOut : Numeric
            Number of photons output from mode.

        Returns
        -------
        None.

        """
        reqDb = 10*np.log10(nIn/nOut)
        
        str1 = 'Photons in: {}\nPhotons out: {}'.format(int(nIn),int(nOut))
        str2 = '\n\nRequired isolation: {}dB'.format(int(reqDb))
        
        self.Label.setText(str1+str2)
        
    
        
        
class QuantumComb:
    def __init__(self,
                 noModes = 100,
                 sigma = 0.5,
                 eta2 = 0.01,
                 power = 0.1,
                 rho = 0.5,
                 Q = 1e8):
        """
        
        This object gives a quantum comb which has all of the parameters
        needed to describe a below-thresholdcomb and calculates the
        spectral parameters associated with it.
        
        All taken from Yanne Chembo's Quantum Dynamics of Kerr Optical
        Frequency Combsbelow and above Threshold:
        Spontaneous Four-Wave-Mixing, Entanglement and Squeezed
        States of Light

        Parameters
        ----------
        noModes : Numeric, optional
            Number of modes to calculate over. The default is 100.
        sigma : Numeric, optional
            Detuning, normalised by linewidth. The default is 0.5.
        eta2 : Numeric, optional
            Dispersion, normalised by linewidth. The default is 0.01.
        power : Numeric, optional
            Cavity power, normalised by linewidth. The default is 0.1.
        rho : Numeric, optional
            Coupling parameter, 0.5 for critical coupling. The default is 0.5.
        Q : Numeric, optional
            Cavity q-factor. The default is 1e8.

        Returns
        -------
        None.

        """
        
        
        self.noModes = noModes # Maximum modal number to simulate
        self.Ls = np.linspace(-self.noModes,self.noModes,
                              2*self.noModes+1) # Mode numbers
        self.sigma = sigma # Detuning, in terms of linewidth 
        self.eta2 = eta2 # Dispersion, in terms of linewidth
        self.rho = rho # Coupling parameter, 1 for add-through
        self.power = power # Coupled power g0|A0|^2, linewidths
        self.Q = Q # Q-factor
        
        self.omegas = np.linspace(-20,20,1000)
        self.calculateSpectra()
        
    def calculateSpectra(self):
        """
        This function calculates the spectra for each mode given
        the parameters. See section IV of the paper.
        """
        self.spectra = []
        self.spectraMax = []
        self.singlePeaked = []
        self.Ns = []
        for l in self.Ls:
            # Eq 63
            etaL = self.sigma - 0.5*self.eta2*l**2 + 2*self.power            
            # Eq 62
            spectrum = 4*self.rho*self.power**2 /(
                (1 - self.power**2 + etaL**2 - self.omegas**2)**2 +
                4*self.omegas**2)
            
            self.spectra.append(spectrum)
            
            # Check whether spectra are single peaked
            if etaL**2<1+self.power:# Eq 66, normalised by kappa
                self.singlePeaked.append(True)
                self.spectraMax.append(4*self.rho*self.power**2/
                                       (1-self.power**2+etaL**2)
                                       **2) # Eq 67
            else:
                self.singlePeaked.append(False)
                self.spectraMax.append(self.rho*self.power**2/
                                       (etaL**2-self.power**2)) # Eq 67
            
            self.kappa = (3e8/1550e-9)/(2*self.Q) # Calculate linewidth
            Rout = (self.rho*self.kappa*(self.kappa**2*self.power**2)/
                    (self.kappa**2*(1-self.power**2+etaL**2))) # Eq 76
            self.Ns.append(Rout)
        
        
    def plotModalEnvelope(self,
                          ax = None,
                          removeLines = True):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        elif removeLines:
            ax.lines = []
        spectraDb = 10*np.log10(self.spectraMax)
        for i, l in enumerate(self.Ls):
            if self.singlePeaked[i]:
                col = 'g'
            else:
                col = 'r'
            ax.plot([l,l], [np.min(spectraDb)-10, spectraDb[i]],
                     col,linewidth=0.75)
        ax.set_ylim([np.min(spectraDb)-10, np.max(spectraDb)+10])
        ax.plot([0,0],[0,0],'g',label='Single Peaked',
                linewidth=0.75)
        ax.plot([0,0],[0,0],'r',label='Double Peaked',
                linewidth=0.75)
        ax.plot(self.Ls,spectraDb,'k',linewidth=2)
        ax.legend()
        ax.set_xlabel('(Relative) Mode Number')
        ax.set_ylabel('Spectral Density (dB)')
        ax.set_title('Envelope of photon flux in different modes')
        
        
    def plotModalSpectrum(self, 
                          mode = 1,
                          ax = None,
                          removeLines = True):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        elif removeLines: 
            ax.lines = []
            
        assert type(mode) == int
        modeIdx = np.where(self.Ls == mode)
        spectrumDb = 10*np.log10(self.spectra[modeIdx[0][0]])
        ax.plot(self.omegas, spectrumDb,'k',
                label='Mode: {}'.format(mode))
        ax.set_ylim([np.min(spectrumDb)-10, np.max(spectrumDb)+10])
        ax.set_xlabel('Frequency, $\omega$/$\kappa$')
        ax.set_title('Spectrum for Mode Number {}'.format(mode))
        
        return self.Ns[modeIdx[0][0]]
        
    def update(self,newdata):
        for key,value in newdata.items():
            setattr(self,key,value)
        self.calculateSpectra()
        
class Resonator:
    """
    This object gives a resonator which can be used to convert experimental 
    parameters to dimensionless quantities for the calculation of photon pair
    production rates
    """
    def __init__(self,
                 n0 = 1.45,
                 d = 660e-6,
                 lam0 = 1550e-9,
                 n2 = 27e-21,
                 Q = 1e8,
                 Aeff = None):
        self.n0 = n0 # Refractive Index
        self.lam0 = lam0 # Wavelength, m
        self.d = d # Diameter,m
        self.n2 = n2 # Nonlinear refractive index, m^2/W - Silica
        self.Q = Q
        
        self.c = 3e8
        self.vg = self.c/self.n0
        self.T_fsr = np.pi*d/self.c
        self.omega0 = 2*np.pi*self.c/self.lam0
        self.hbar = 1.05e-34 
        
        if Aeff == None:
            # Modal area for spherical resonator, worst case scenario, 
            self.Aeff = (self.lam0/self.n0)**(7/6) * (self.d/2)**(5/6)
        else:
            self.Aeff = Aeff
        
        self.g0 = ((self.omega0*self.n2*self.vg*self.hbar*self.omega0)/ 
                   (self.c*self.Aeff*self.T_fsr))
        
    def convertCavityPhotonNumber(self,
                                  power = 0.0):
        self.photonNumber = power/self.g0 # Power given as g0 |A0|^2, in
        # terms of linewidths 
        
        # For cavity Q = omega0*E/P, where Q is the Q-factor, omega0 is the 
        # angular frequency, E is the energy stored in the resonator (i.e. N
        # * hBar * omega0 where N is the cavity photon number) and P is the power 
        # dissipated. At critical coupling & steady state, this P = P_in = 
        # N_in * hbar * omega0. So the input photon flux per second is:
        self.N_in = self.photonNumber * self.omega0/self.Q
        
class Gui(QtWidgets.QMainWindow):
    """
    This object gives a GUI that can be used to interact with the quantum comb
    simulator to calculate photon flux/spectra 
    """
    def __init__(self):
        super(Gui,self).__init__()
        self.setWindowTitle('Quantum Comb Spectra')
        self.setFixedHeight(1500)
        self.setFixedWidth(3000)
        
        self.QuantumComb = QuantumComb()
        self.Resonator = Resonator()
        self.Resonator.convertCavityPhotonNumber(self.QuantumComb.power*
                                                 self.QuantumComb.kappa)
        self.nIn = self.Resonator.N_in
        
        self.envelopeWidget = MplCanvas()        
        self.spectrumWidget = MplCanvas()
        
        self.makeEnvelope()
        self.makeSpectrum()
        self.makeOptions()
        
        self.hbox = QtWidgets.QHBoxLayout()
        self.hbox.addWidget(self.envelopeWidget)
        self.hbox.addWidget(self.spectrumWidget)
        self.hbox.addWidget(self.optionsWidget)
        
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(self.hbox)
        self.setCentralWidget(self.centralWidget)
        
    def makeEnvelope(self):
        self.QuantumComb.plotModalEnvelope(ax = self.envelopeWidget.axes)
        self.envelopeWidget.mpl_connect('button_press_event', 
                                        self.updateSpectrum)
        self.envelopeWidget.draw()
        self.show()
        
    def makeSpectrum(self,
                     mode = 1):
        nOut = self.QuantumComb.plotModalSpectrum(ax =self.spectrumWidget.axes,
                                                  mode = mode)
        self.nOut = nOut
        self.spectrumWidget.draw()
        
        self.show()
        
    def makeOptions(self):
        
        self.optionsWidget = QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.addItem(
            QtWidgets.QSpacerItem(20,
                                  40, 
                                  QtWidgets.QSizePolicy.Minimum,
                                  QtWidgets.QSizePolicy.Expanding))
        layout.addWidget(MyLabel(name='Power',
                                 value = self.QuantumComb.power,
                                 varName = 'power',
                                 parent = self))
        layout.addWidget(MyLabel(name='Detuning',
                                 value = self.QuantumComb.sigma,
                                 varName = 'sigma',
                                 parent = self))
        layout.addWidget(MyLabel(name='Dispersion',
                                 value = self.QuantumComb.eta2,
                                 varName = 'eta2',
                                 parent = self))
        layout.addWidget(MyLabel(name=('Out coupling (0.5 or 1)'),
                                 value = self.QuantumComb.rho,
                                 varName = 'rho',
                                 parent = self))
        layout.addItem(
            QtWidgets.QSpacerItem(20,
                                  40, 
                                  QtWidgets.QSizePolicy.Minimum,
                                  QtWidgets.QSizePolicy.Expanding))
        
        self.myDisplay = MyDisplayWidget(nIn = self.nIn, nOut = self.nOut)
        layout.addWidget(self.myDisplay)
        
        self.optionsWidget.setLayout(layout)
        
    def updateSpectrum(self,event):
        mode = np.round(event.xdata)
        self.makeSpectrum(mode=int(mode))
        self.myDisplay.updateDisplay(self.nIn, self.nOut)
        
    def updateOptionValue(self, option = None, value = None):
        newdata = {option:value}
        self.QuantumComb.update(newdata)
        self.Resonator.convertCavityPhotonNumber(self.QuantumComb.power)
        self.nIn = self.Resonator.N_in
        self.myDisplay.updateDisplay(self.nIn, self.nOut)
        self.makeEnvelope()
        self.makeSpectrum(1)
        self.show()
        
def main():  
    app = QApplication(sys.argv)
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    x = Gui()
    x.show()
    sys.exit(app.exec_())
            
if __name__ == '__main__':
    main()