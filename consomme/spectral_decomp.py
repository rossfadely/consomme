import os
import numpy as np


class Consumme(object):

    def __init__(self,spectralist):

        self.load_data(spectralist)
        self.make_default_grid()
        
    def load_data(self,spectralist):
        """
        Load in the SDSS spectral data. 
        """
        f = open(spectralist)
        lines = f.readlines()
        f.close()

        self.N = len(lines)
        for i in range(self.N):
            f = pf.open(lines[i][:-1])
            d1 = f[1].data
            d2 = f[2].data            
            if i==0:
                self.D = d1[0].field(loglam).shape[0]
                self.data = np.zeros(self.N,self.D)
                self.ivar = self.data.copy()
                self.wave = self.data.copy()
                self.z    = np.zeros(self.N)
            else:
                assert d1[0].field(loglam).shape[0]==self.D
            self.data[i] = d1.field('flux')
            self.ivar[i] = d1.field('ivar')
            self.wave[i] = 10.0 ** d1.field('loglam'))
            self.z[i]    = d2.field('z')
  
    def make_default_grid(self):

        ind = np.where(self.z == np.median(self.z))[0]
        default_grid = self.wave[ind] / (1. + self.z[ind])
        print default_grid,z
