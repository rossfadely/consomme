import os
import numpy as np
import pyfits as pf
import matplotlib.pyplot as pl

from interpolator import *

class SDSSSpecPrep(object):
    """
    Prepare SDSS spectra for factorization.
    """
    def __init__(self,spectralist,dlt=5.e-5,
                 normalize=True,wavelimits=(5000,6000),
                 ivar_factor=100,ivar_cut=0.001):

        self.load_data(spectralist)
        self.make_default_grid(dlt)
        self.interpolate('default')
        if normalize:
            self.normalize(wavelimits[0],wavelimits[1])
        self.set_ivar_and_flux(ivar_cut,ivar_factor)
        
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
            f.close()
            if i==0:
                # grid size of SDSS data not 
                # always the same
                self.flux = {}
                self.ivar = {}
                self.wave = {}
                self.z    = np.zeros(self.N)
            self.flux[i] = d1.field('flux')
            self.ivar[i] = d1.field('ivar')
            self.wave[i] = 10.0 ** (d1.field('loglam'))
            self.z[i]    = d2.field('z')
  
    def make_default_grid(self,dlt):
        """
        Construct the final wavelength grid.
        """
        mx = 0
        mn = np.Inf
        for i in range(self.N):
            self.wave[i] /= (1. + self.z[i])
            if self.wave[i][0] < mn: 
                mn = self.wave[i][0]
            if self.wave[i][-1] > mx: 
                mx = self.wave[i][-1]
        mn = np.log10(mn) - dlt
        mx = np.log10(mx) + dlt

        def_grid = np.array([mn])
        lam = mn
        while lam<mx:
            lam += dlt
            def_grid = np.append(def_grid,lam)
        
        self.final_wave = 10.**def_grid[:-1]
        self.D = self.final_wave.shape[0]
        
    def interpolate(self,method):
        """
        Interpolate onto the final grid.
        """
        self.final_flux = np.zeros((self.N,self.D))
        self.final_ivar = np.zeros((self.N,self.D))

        if method!='scipy':
            self.final_wave = self.final_wave.astype('float64')
            self.final_flux = self.final_flux.astype('float64')
            self.final_ivar = self.final_ivar.astype('float64')

            interp = get_interp('/home/rfadely/local/lib/',
                                './_cubic_spline_interp_1d.so')


            for i in range(self.N):
                mask = interpolate_spectrum(interp,
                                            self.wave[i].astype('float64'),
                                            self.flux[i].astype('float64'),
                                            self.final_wave,
                                            self.final_flux[i,:])
                mask = interpolate_spectrum(interp,
                                            self.wave[i].astype('float64'),
                                            self.ivar[i].astype('float64'),
                                            self.final_wave,
                                            self.final_ivar[i,:])

                # zero ivars for points outside interpolation range
                ind = mask == 1.
                self.final_ivar[i,ind] = 0.0
                # zero ivars for places where interp goes negative
                ind = self.final_ivar[i,:] < 0.0
                self.final_ivar[i,ind] = 0.0

        ind = np.where(np.sum(self.final_ivar,axis=0) != 0)[0]

        self.final_wave = self.final_wave[ind]
        self.final_flux = self.final_flux[:,ind]
        self.final_ivar = self.final_ivar[:,ind]
        self.D = self.final_wave.shape[0]

    def normalize(self,wavemin,wavemax):
        """
        Normalize the spectra using a weighted mean in a 
        specified interval.
        """
        ind = np.where((self.final_wave>wavemin) & (self.final_wave<wavemax))[0]
        means = np.sum(self.final_flux[:,ind]*self.final_ivar[:,ind],axis=1) / \
            np.sum(self.final_ivar[:,ind],axis=1)

        self.final_flux /= means[:,None]
        self.final_ivar /= means[:,None]

    def set_ivar_and_flux(self,ivar_cut,ivar_factor):
        """
        Infill places in spectrum with invar = 0, and 
        replace invars with a large factor.
        """
        mean = np.sum(self.final_flux*self.final_ivar,axis=0) / \
            np.sum(self.final_ivar,axis=0)

        for i in range(self.D):
            ind = np.where(self.final_ivar[:,i]<ivar_cut)[0]
            self.final_ivar[ind,i] = 0.0
            ind = np.where(self.final_ivar[:,i]==0.0)[0]
            self.final_flux[ind,i] = mean[i]
            self.final_ivar[ind,i] = ivar_factor


