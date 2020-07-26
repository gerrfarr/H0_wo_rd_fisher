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
            omega_b=0.022032,
            omega_cdm=0.12038,
            n_s=0.9667,
            A_s=2.0989e-9,
            sound_horizon_scaling=1.0,
            N_eff=3.046):

        self.h=h
        self.T0_cmb=T0_cmb
        self.omega_b=omega_b
        self.omega_cdm=omega_cdm
        self.n_s=n_s
        self.A_s=A_s
        self.sound_horizon_scaling=sound_horizon_scaling
        self.N_eff=N_eff

        self.__class = Class()
        self.__computed = False

    @property
    def Omega0_m(self):
        return self.Omega0_b+self.Omega0_cdm

    @property
    def Omega0_b(self):
        return self.omega_b/self.h**2

    @property
    def Omega0_cdm(self):
        return self.omega_cdm/self.h**2

    @property
    def sigma8(self):
        return self.__class.sigma8()

    @property
    def norm(self):
        return self.A_s/A_s_norm

    @norm.setter
    def norm(self, value):
        self.A_s = value*A_s_norm

    @property
    def computed(self):
        return self.__computed

    def scale_independent_growth_factor(self, redshift):
        return self.__class.scale_independent_growth_factor(redshift)

    @property
    def class_params(self):
        return {
            'A_s': self.A_s,
            'n_s': self.n_s,
            'T_cmb': self.T0_cmb,
            'h': self.h,
            'omega_b': self.omega_b,
            'omega_cdm': self.omega_cdm,
            'N_ur': self.N_eff,
            'output': 'mPk'
        }

    def get_class(self):
        return self.__class

    def compute(self, force=True):
        if self.__computed and force:
            self.__class.empty()
        elif self.__computed and not force:
            print("Class has already computed all relevant properties.")
            return

        self.__class.set(self.class_params)
        self.__class.compute()
        self.__computed = True

    def clone(self):
        return Cosmology(
            h=self.h,
            T0_cmb=self.T0_cmb,
            omega_b=self.omega_b,
            omega_cdm=self.omega_cdm,
            n_s=self.n_s,
            A_s=self.A_s,
            sound_horizon_scaling=self.sound_horizon_scaling,
            N_eff=self.N_eff)
