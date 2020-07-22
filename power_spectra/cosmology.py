"""
this code provides basic cosmological parameters.
It is designed to emulate the Cosmology class from nbodykit (https://github.com/bccp/nbodykit) while reducing complexity.

Jul 2020
"""
from classy import Class
A_s_norm=2.0989e-9
class Cosmology(object):
    def __init__(self,
            h=0.67556,
            T0_cmb=2.7255,
            Omega0_b=0.022032/0.67556**2,
            Omega0_cdm=0.12038/0.67556**2,
            n_s=0.9667,
            A_s=2.0989e-9,
            alpha_rs=1.0,
            N_eff=3.046):

        self.h=h
        self.T0_cmb=T0_cmb
        self.Omega0_b=Omega0_b
        self.Omega0_cdm=Omega0_cdm
        self.n_s=n_s
        self.A_s=A_s
        self.alpha_rs=alpha_rs
        self.N_eff=N_eff

        self.__class = Class()
        self.__class_params = {
        'A_s':self.A_s,
        'n_s':self.n_s,
        'T_cmb':self.T0_cmb,
        'h':self.h,
        'omega_b':self.omega_b,
        'omega_cdm':self.omega_cdm,
        'N_ur':self.N_eff,
        'output':'mPk'
        }
        self.__class.set(self.__class_params)
        self.__class.compute()

    @property
    def Omega0_m(self):
        return self.Omega0_b+self.Omega0_cdm

    @property
    def omega_b(self):
        return self.Omega0_b*self.h**2

    @property
    def omega_cdm(self):
        return self.Omega0_cdm*self.h**2

    @property
    def sigma8(self):
        return self.__class.sigma8()

    @property
    def norm(self):
        return self.A_s/A_s_norm

    def scale_independent_growth_factor(self, redshift):
        return self.__class.scale_independent_growth_factor(redshift)

    def class_params(self):
        return self.__class_params

    def get_class(self):
        return self.__class