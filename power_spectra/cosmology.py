"""
this code provides basic cosmological parameters.
It is designed to emulate the Cosmology class from nbodykit (https://github.com/bccp/nbodykit) while reducing complexity.

There are two classes provided. "Cosmology" provides a simplified cosmological model without massive neutrinos that can be used with the Eisenstein-Hu transfer functiosn to obtain analytical power spectra. "ClassCosmology" takes as input parameters for the boltzmann solver Class and allows to vary all cosmological parameters implemented in Class.

Jul 2020
edited: Dec 2020
Gerrit Farren
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

        self._class = Class()
        self._class.set(self.class_params)
        self._computed = False

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
        return self._class.sigma8()

    @property
    def norm(self):
        return (self.A_s/A_s_norm)**(1/2)

    @norm.setter
    def norm(self, value):
        self.A_s = value**2.0*A_s_norm

    @property
    def computed(self):
        return self._computed

    def scale_independent_growth_factor(self, redshift):
        return self._class.scale_independent_growth_factor(redshift)

    def scale_independent_growth_factor_f(self, redshift):
        return self._class.scale_independent_growth_factor_f(redshift)

    @property
    def class_params(self):
        return {
            'A_s': self.A_s,
            'n_s': self.n_s,
            'T_cmb': self.T0_cmb,
            'h': self.h,
            'omega_b': self.omega_b,
            'omega_cdm': self.omega_cdm,
            'N_ncdm':0,
            'N_ur':self.N_eff
        }

    def get_class(self):
        return self._class

    def compute(self, force=True):
        if self._computed and force:
            self._class.empty()
        elif self._computed and not force:
            print("Class has already computed all relevant properties.")
            return

        self._class.compute()
        self._computed = True

    def clone(self, pre_computed=False):
        if pre_computed and self._computed:
            return self
        elif pre_computed and not self._computed:
            self.compute()
            return self
        else:
            return Cosmology(
                h=self.h,
                T0_cmb=self.T0_cmb,
                omega_b=self.omega_b,
                omega_cdm=self.omega_cdm,
                n_s=self.n_s,
                A_s=self.A_s,
                sound_horizon_scaling=self.sound_horizon_scaling,
                N_eff=self.N_eff)

class ClassCosmology(Cosmology):
    def __init__(self, class_params):

        self.__class_params=class_params

        self._class = Class()
        self._class.set(self.class_params)
        self._computed = False

    @property
    def h(self):
        return self._class.h()

    @property
    def T0_cmb(self):
        return self._class.T_cmb()

    @property
    def A_s(self):
        return self._class.A_s()

    @property
    def n_s(self):
        return self._class.n_s()

    @property
    def Omega0_b(self):
        return self._class.Omega_b()

    @property
    def Omega0_cdm(self):
        return self._class.omegach2()/self.h**2

    @property
    def Omega0_m(self):
        return self._class.Omega0_m()

    @property
    def norm(self):
        return (self.A_s / A_s_norm)**(1 / 2)

    @norm.setter
    def norm(self, value):
        NotImplementedError("This function is not implemented here.")

    @property
    def class_params(self):
        return self.__class_params

    def compute(self, force=True):
        if self._computed and force:
            self._class.empty()
        elif self._computed and not force:
            print("Class has already computed all relevant properties.")
            return

        self._class.compute()
        self._computed = True

    def clone(self, pre_computed=False):
        if pre_computed and self._computed:
            return self
        elif pre_computed and not self._computed:
            self.compute()
            return self
        else:
            return ClassCosmology(self.class_params)
