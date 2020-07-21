"""
this code provides basic cosmological parameters.
It is designed to emulate the Cosmology class from nbodykit (https://github.com/bccp/nbodykit) without the connectivity to CLASS and some additional parameters.

Jul 2020
"""
import numpy
from six import string_types
import os
import functools
from classy import Class

class Cosmology(object):
    def __init__(self,
            h=0.67556,
            T0_cmb=2.7255,
            Omega0_b=0.022032/0.67556**2,
            Omega0_cdm=0.12038/0.67556**2,
            n_s=0.9667,
            A_s=2.0989e-9,
            alpha_rs=1,
            N_eff=3.046):
        self.h=h
        self.T0_cmb=T0_cmb
        self.Omega0_b=Omega0_b
        self.Omega0_cdm=Omega0_cdm
        self.n_s=n_s
        self.A_s=A_s
        self.alpha_rs=alpha_rs
        self.N_eff=N_eff

        self._class = Class()
        class_params = {
        'A_s':A_s,
        'n_s':n_s,
        'T_cmb':T0_cmb,
        'h':h,
        'omega_b':self.omega_b,
        'omega_cdm':self.omega_cdm,
        'N_ur':self.N_eff,
        'output':'mPk'
        }
        self._class.set(class_params)
        self._class.compute()

    @property
    def Omega0_m(self):
        return self.Omega0_b+self.Omega0_cdm

    @property
    def omega_b(self):
        return self.Omega0_b*self.h**2

    @property
    def omega_cdm(self):
        return self.Omega0_cdm*self.h**2

    def scale_independent_growth_factor(self, redshift):
        return self._class.scale_independent_growth_factor(redshift)

    @property
    def sigma8(self):
        return self._class.sigma8()