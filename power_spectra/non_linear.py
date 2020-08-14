import tempfile
from copy import copy
from classy import Class
import _warnings as warnings
import numpy as np

from .linear_power import LinearPower
from .cosmology import Cosmology
from ..custom_exceptions import ClassComputationError, OrderOfOperationsError, ParameterValueError, TemporaryFileClosedWarning


class BiasParams(object):
    def __init__(self, b1=2.0, b2=0.0, bG2=0.0, css0=0.0, css2=0.0, b4=500.0, Pshot=3500.0, bGamma3=0.0):
        self.b1 = b1
        self.b2 = b2
        self.bG2 = bG2
        self.css0 = css0
        self.css2 = css2
        self.b4 = b4
        self.Pshot = Pshot
        self.bGamma3 = bGamma3

    def clone(self):
        return BiasParams(b1=self.b1, b2=self.b2, bG2=self.bG2, css0=self.css0, css2=self.css2, b4=self.b4, Pshot=self.Pshot, bGamma3=self.bGamma3)

class ParameterPackage(object):
    cosmo_param_names = ['h', 'T0_cmb', 'n_s', 'A_s', 'sound_horizon_scaling', 'N_eff', 'norm', 'omega_b', 'omega_cdm']
    bias_param_names = ['b1', 'b2', 'bG2', 'css0', 'css2', 'b4', 'Pshot', 'bGamma3']
    def __init__(self, cosmo, bias_params):
        """
        Packages all parameters for the cosmology and BiasParams object.

        Parameters
        ----------
        cosmo : Cosmology
        bias_params : BiasParams
        """
        self.__cosmo = cosmo
        self.__bias_params = bias_params

    def __setattr__(self, key, value):
        """
        Overwritten setattr
        Parameters
        ----------
        key
        value

        Returns
        -------

        """
        if key in ['_ParameterPackage__cosmo', '_ParameterPackage__bias_params']:
            super().__setattr__(key, value)
        elif key in self.cosmo_param_names:
            setattr(self.__cosmo, key, value)
        elif key in self.bias_param_names:
            setattr(self.__bias_params, key, value)
        else:
            raise AttributeError("Set:Neither Cosmology nor BiasParams have an attribute named {}.".format(key))

    def __getattr__(self, key):
        if key in self.cosmo_param_names or key in ['class_params', 'get_class', 'compute', 'computed', 'Omega0_m', 'scale_independent_growth_factor', 'sigma8', 'Omega0_b', 'Omega0_cdm']:
            return getattr(self.__cosmo, key)
        elif key in self.bias_param_names:
            return getattr(self.__bias_params, key)
        else:
            raise AttributeError("Get:Neither Cosmology nor BiasParams have an attribute named {}.".format(key))

    def clone(self, pre_computed=False):
        cosmo_clone=self.__cosmo.clone(pre_computed)
        bias_params_clone=self.__bias_params.clone()

        return ParameterPackage(cosmo_clone, bias_params_clone)

    @property
    def bias_params(self):
        return self.__bias_params

    @property
    def cosmo(self):
        return self.__cosmo

    @staticmethod
    def in_cosmo(param_name):
        return param_name in ParameterPackage.cosmo_param_names

    @staticmethod
    def in_bias_params(param_name):
        return param_name in ParameterPackage.bias_param_names



class NonLinearPower(object):
    def __init__(self, cosmo, linPower, redshift, k_vals_h_invMpc=None):
        """

        Parameters
        ----------
        cosmo : Cosmology
        linPower : LinearPower
        redshift : float
        k_vals_h_invMpc : numpy.ndarray
        """
        self.__cosmo = cosmo
        self.__linPower = linPower
        self.__redshift = redshift
        self.__computed = False

        if k_vals_h_invMpc is None:
            k_vals_h_invMpc = np.logspace(-5, 3, 10000)

        pk_lin_vals = linPower(k_vals_h_invMpc, redshift)

        self.__temporary_power_spectrum_file = tempfile.NamedTemporaryFile()

        np.savetxt(self.get_power_spectrum_path(), np.vstack([k_vals_h_invMpc * cosmo.h, pk_lin_vals / cosmo.h ** 3 / cosmo.norm]).T, delimiter='\t')

        self.__class = Class()
        self.__class.set(cosmo.class_params)

        class_non_linear_params = {'z_pk': redshift,
                                   'non linear': ' PT ',
                                   'IR resummation': ' Yes ',
                                   'Bias tracers': ' Yes ',
                                   'RSD': ' Yes ',
                                   'AP': 'No',
                                   'FFTLog mode': 'FAST',
                                   'Input Pk': self.get_power_spectrum_path(),
                                   }

        self.__class.set(class_non_linear_params)

    def get_non_linear_class(self):
        return self.__class

    @property
    def computed(self):
        return self.__computed

    def compute(self):
        if not self.__computed:
            try:
                self.__class.compute()
            except Exception as ex:
                print("Got the following error while computing non-linear power spectra:")
                print(str(ex))
                raise ClassComputationError("Non-linear power spectra computation failed.")
            self.__computed = True
            self.__temporary_power_spectrum_file.close()
        else:
            print("Non-linear power spectra already available.")

    def get_power_spectrum_path(self):
        if not self.__temporary_power_spectrum_file.closed:
            return self.__temporary_power_spectrum_file.name
        else:
            warnings.warn("The power spectrum file has been closed and removed.", TemporaryFileClosedWarning)
            return None

    def get_temp_file(self):
        if not self.__temporary_power_spectrum_file.closed:
            return self.__temporary_power_spectrum_file
        else:
            warnings.warn("The power spectrum file has been closed and removed.", TemporaryFileClosedWarning)
            return None

    def get_non_linear(self, ell, kbins, bias_params):
        """

        Parameters
        ----------
        ell : int
            Select Mono- or Quadrupole
        bias_params : BiasParams
        kbins : array_like

        Returns
        ----------
        array_like
            Biased non-linear power spectra (Monopole and Quandrupole depending on value of `ell`)
        """

        if not self.__computed:
            raise OrderOfOperationsError("Non-linear power spectra can not be obtained. They have not been computed yet.")
        else:
            if ell == 2:
                fz = self.__class.scale_independent_growth_factor_f(self.__redshift)
                pk_mult = self.__class.get_pk_mult(kbins * self.__cosmo.h, self.__redshift, len(kbins))

                return (self.__cosmo.norm ** 2. * pk_mult[18] \
                        + self.__cosmo.norm ** 4. * (pk_mult[24]) \
                        + self.__cosmo.norm ** 1. * bias_params.b1 * pk_mult[19] + self.__cosmo.norm ** 3. * bias_params.b1 * (pk_mult[25]) \
                        + bias_params.b1 ** 2. * self.__cosmo.norm ** 2. * pk_mult[26] \
                        + bias_params.b1 * bias_params.b2 * self.__cosmo.norm ** 2. * pk_mult[34] \
                        + bias_params.b2 * self.__cosmo.norm ** 3. * pk_mult[35] \
                        + bias_params.b1 * bias_params.bG2 * self.__cosmo.norm ** 2. * pk_mult[36] \
                        + bias_params.bG2 * self.__cosmo.norm ** 3. * pk_mult[37] \
                        + 2. * (bias_params.css2 + 0. * bias_params.b4 * kbins ** 2. * self.__cosmo.h ** 2.) * self.__cosmo.norm ** 2. * pk_mult[12] / self.__cosmo.h ** 2. \
                        + (2. * bias_params.bG2 + 0.8 * bias_params.bGamma3) * self.__cosmo.norm ** 3. * pk_mult[9]) * self.__cosmo.h ** 3. \
                       + fz ** 2. * bias_params.b4 * kbins ** 2. * ((self.__cosmo.norm ** 2. * fz ** 2. * 70. + 165. * fz * bias_params.b1 * self.__cosmo.norm + 99. * bias_params.b1 ** 2.) * 4. / 693.) * (35. / 8.) * pk_mult[13] * self.__cosmo.h

            elif ell == 0:
                fz = self.__class.scale_independent_growth_factor_f(self.__redshift)
                pk_mult = self.__class.get_pk_mult(kbins * self.__cosmo.h, self.__redshift, len(kbins))
                return (self.__cosmo.norm ** 2. * pk_mult[15] \
                        + self.__cosmo.norm ** 4. * pk_mult[21] \
                        + self.__cosmo.norm ** 1. * bias_params.b1 * pk_mult[16] \
                        + self.__cosmo.norm ** 3. * bias_params.b1 * pk_mult[22] \
                        + self.__cosmo.norm ** 0. * bias_params.b1 ** 2. * pk_mult[17] \
                        + self.__cosmo.norm ** 2. * bias_params.b1 ** 2. * pk_mult[23] \
                        + 0.25 * self.__cosmo.norm ** 2. * bias_params.b2 ** 2. * pk_mult[1] \
                        + bias_params.b1 * bias_params.b2 * self.__cosmo.norm ** 2. * pk_mult[30] \
                        + bias_params.b2 * self.__cosmo.norm ** 3. * pk_mult[31] \
                        + bias_params.b1 * bias_params.bG2 * self.__cosmo.norm ** 2. * pk_mult[32] \
                        + bias_params.bG2 * self.__cosmo.norm ** 3. * pk_mult[33] \
                        + bias_params.b2 * bias_params.bG2 * self.__cosmo.norm ** 2. * pk_mult[4] \
                        + bias_params.bG2 ** 2. * self.__cosmo.norm ** 2. * pk_mult[5] \
                        + 2. * bias_params.css0 * self.__cosmo.norm ** 2. * pk_mult[11] / self.__cosmo.h ** 2. \
                        + (2. * bias_params.bG2 + 0.8 * bias_params.bGamma3) * self.__cosmo.norm ** 2. * (bias_params.b1 * pk_mult[7] + self.__cosmo.norm * pk_mult[8])) * self.__cosmo.h ** 3. \
                       + bias_params.Pshot \
                       + fz ** 2. * bias_params.b4 * kbins ** 2. * (self.__cosmo.norm ** 2. * fz ** 2. / 9. + 2. * fz * bias_params.b1 * self.__cosmo.norm / 7. + bias_params.b1 ** 2. / 5) * (35. / 8.) * pk_mult[13] * self.__cosmo.h

            else:
                raise ParameterValueError("The parameter ell has taken the unsupported value {}.".format(ell))
