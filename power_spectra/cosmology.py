"""
this code provides basic cosmological parameters.
It is designed to emulate the Cosmology class from nbodykit (https://github.com/bccp/nbodykit) while reducing complexity.

There are two classes provided. "Cosmology" provides a simplified cosmological model without massive neutrinos that can be used with the Eisenstein-Hu transfer functiosn to obtain analytical power spectra. "ClassCosmology" takes as input parameters for the boltzmann solver Class and allows to vary all cosmological parameters implemented in Class.

Jul 2020
edited: Dec 2020
Gerrit Farren
"""
from classy_pt import Class
import numpy as np
from copy import copy

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
            peak_amp_scaling=1.0,
            suppression_scaling=1.0,
            N_eff=3.046):

        self.h=h
        self.T0_cmb=T0_cmb
        self.omega_b=omega_b
        self.omega_cdm=omega_cdm
        self.n_s=n_s
        self.A_s=A_s
        self.sound_horizon_scaling=sound_horizon_scaling
        self.suppression_scaling = suppression_scaling
        self.peak_amp_scaling = peak_amp_scaling
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

    def Hubble(self, z):
        return self._class.Hubble(z)

    def angular_distance(self, z):
        return self._class.angular_distance(z)

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

    def z_of_r(self, z):
        return self._class.z_of_r(z)

    def rs_drag(self):
        return self.sound_horizon_scaling*self._class.rs_drag()

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
                peak_amp_scaling=self.peak_amp_scaling,
                suppression_scaling=self.suppression_scaling,
                N_eff=self.N_eff)

class ClassCosmology(Cosmology):
    def __init__(self, class_params,  ClassVersion=None, automatically_recompute=True):

        self.__class_params=class_params

        if ClassVersion is None:
            self.__ClassVersion = None
            self._class = Class()
        else:
            self.__ClassVersion = ClassVersion
            self._class = ClassVersion()

        self._class.set(self.__class_params)
        self._computed = False
        self.__automatically_recompute=automatically_recompute

        self.sound_horizon_scaling = 1.0

    @property
    def T0_cmb(self):
        return self._class.T_cmb()

    @property
    def A_s(self):
        try:
            return float(self.__class_params['A_s'])
        except KeyError:
            return np.exp(float(self.__class_params['ln10^{10}A_s']))/1.0e10

    @A_s.setter
    def A_s(self, new):
        self.__class_params['A_s'] = new
        if self.__automatically_recompute:
            self.compute(force=True)

    @property
    def n_s(self):
        return self._class.n_s()

    @n_s.setter
    def n_s(self, new):
        self.__class_params['n_s'] = new
        if self.__automatically_recompute:
            self.compute(force=True)

    @property
    def Omega0_b(self):
        return self._class.Omega_b()

    @property
    def Omega0_cdm(self):
        return self._class.Omega0_cdm()

    @property
    def Omega0_m(self):
        return self._class.Omega0_m()

    @property
    def peak_amp_scaling(self):
        return 1.0

    @property
    def suppression_scaling(self):
        return 1.0

    @property
    def norm(self):
        return (self.A_s / A_s_norm)**(1 / 2)

    @norm.setter
    def norm(self, value):
        NotImplementedError("This function is not implemented here.")

    @property
    def omega_cdm(self):
        return self.__class_params['omega_cdm']

    @omega_cdm.setter
    def omega_cdm(self, new):
        self.__class_params['omega_cdm'] = new
        if self.__automatically_recompute:
            self.compute(force=True)

    @property
    def h(self):
        return self.__class_params['h']

    @h.setter
    def h(self, new):
        self.__class_params['h'] = new
        if self.__automatically_recompute:
            self.compute(force=True)

    @property
    def omega_b(self):
        return self.__class_params['omega_b']

    @omega_b.setter
    def omega_b(self, new):
        self.__class_params['omega_b'] = new
        if self.__automatically_recompute:
            self.compute(force=True)

    @property
    def class_params(self):
        return self.__class_params

    def compute(self, force=True):
        if self._computed and force:
            self._class.empty()
        elif self._computed and not force:
            print("Class has already computed all relevant properties.")
            return

        self._class.set(self.__class_params)
        self._class.compute()
        self._computed = True

    def clone(self, pre_computed=False):
        if pre_computed and self._computed:
            return self
        elif pre_computed and not self._computed:
            self.compute()
            return self
        else:
            return ClassCosmology(copy(self.__class_params), self.__ClassVersion)

    def copy_as_non_class(self):
        return Cosmology(
            h=self.h,
            T0_cmb=self.T0_cmb,
            omega_b=self.omega_b,
            omega_cdm=self.omega_cdm,
            n_s=self.n_s,
            A_s=self.A_s,
            sound_horizon_scaling=self.sound_horizon_scaling,
            peak_amp_scaling=self.peak_amp_scaling,
            suppression_scaling=self.suppression_scaling)
