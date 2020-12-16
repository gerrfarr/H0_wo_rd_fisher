"""
This code provides a basic wrapper to Class-PT.

The class BiasParams provides a container for EFTofLSS bias parameters. Those are packaged together and made available in a convenient way with the cosmological parameters in ParameterPackage. The two classes NonLinearPower and NonLinearPowerReplace give the ability to invoke Class-PT with a given cosmology. In NonLinearPowerReplace one additionally has the ability to replace the input linear power spectrum with a custom power spectrum provided as an input. This allows to compute non-linear power spectra for models not implemented in Class.

Jul 2020
edited: Dec 2020
Gerrit Farren
"""

import tempfile
from copy import copy
from classy import Class
import _warnings as warnings
import numpy as np

from .linear_power import LinearPower
from .cosmology import Cosmology
from ..custom_exceptions import ClassComputationError, OrderOfOperationsError, ParameterValueError, TemporaryFileClosedWarning


class BiasParams(object):
    def __init__(self, b1=0.0, b2=0.0, bG2=0.0, css0=0.0, css2=0.0, css4=0.0, b4=0.0, Pshot=0.0, bGamma3=0.0, a0=0.0, a2=0.0):
        self.b1 = b1
        self.b2 = b2
        self.bG2 = bG2
        self.css0 = css0
        self.css2 = css2
        self.css4 = css4
        self.b4 = b4
        self.Pshot = Pshot
        self.bGamma3 = bGamma3
        self.a0 = a0
        self.a2 = a2

    def clone(self):
        return BiasParams(b1=self.b1, b2=self.b2, bG2=self.bG2, css0=self.css0, css2=self.css2, css4=self.css4, b4=self.b4, Pshot=self.Pshot, bGamma3=self.bGamma3, a0=self.a0, a2=self.a2)

    def print(self):
        print(f"b1={self.b1}, b2={self.b2}, bG2={self.bG2}, css0={self.css0}, css2={self.css2}, css4={self.css4}, b4={self.b4}, Pshot={self.Pshot}, bGamma3={self.bGamma3}, a0={self.a0}, a2={self.a2}")

    def get_tuple(self):
        return self.b1, self.b2, self.bG2, self.css0, self.css2, self.css4, self.b4, self.Pshot, self.bGamma3, self.a0, self.a2

class ParameterPackage(object):
    cosmo_param_names = ['h', 'T0_cmb', 'n_s', 'A_s', 'sound_horizon_scaling', 'N_eff', 'norm', 'omega_b', 'omega_cdm']
    bias_param_names = ['b1', 'b2', 'bG2', 'css0', 'css2', 'css4', 'b4', 'Pshot', 'bGamma3', 'a0', 'a2']
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
    def __init__(self, cosmo, redshift, Omfid=None, renormalize=True, no_wiggle=False):
        """

        Parameters
        ----------
        no_wiggle : bool
        renormalize : bool
        Omfid : float
        cosmo : Cosmology
        redshift : float
        """
        self.__cosmo = cosmo
        self.__redshift = redshift
        self.__computed = False

        if Omfid is None:
            Omfid=cosmo.Omega0_m

        if renormalize:
            self.__norm = cosmo.norm
        else:
            self.__norm = 1.0

        self.__class = Class()
        self.__class.set(cosmo.class_params)

        class_non_linear_params = {'output': "mPk",
                                   'z_pk': redshift,
                                   'non linear': ' PT ',
                                   'IR resummation': ' Yes ',
                                   'Bias tracers': ' Yes ',
                                   'RSD': ' Yes ',
                                   'AP': 'Yes',
                                   'output format': 'FAST',
                                   'FFTLog mode': 'FAST',
                                   'Omfid': Omfid,
                                   'A_s': 2.0989e-9 if renormalize else self.__cosmo.A_s,
                                   'no-wiggle': 'YES' if no_wiggle else 'NO',
                                   'P_k_max_h/Mpc': '100.',
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
        else:
            print("Non-linear power spectra already available.")


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
            if ell == 4:
                fz = self.__class.scale_independent_growth_factor_f(self.__redshift)
                pk_mult = self.__class.get_pk_mult(kbins * self.__cosmo.h, self.__redshift, len(kbins))
                return ((self.__norm**2 * pk_mult[20]
                            + self.__norm**4 * pk_mult[27]
                            + bias_params.b1 * self.__norm**3 * pk_mult[28]
                            + bias_params.b1**2 * self.__norm**2 * pk_mult[29]
                            + bias_params.b2 * self.__norm**3 * pk_mult[38]
                            + bias_params.bG2 * self.__norm**3 * pk_mult[39]
                            + 2. * bias_params.css4 * self.__norm**2 * pk_mult[13] / self.__cosmo.h**2) * self.__cosmo.h**3
                        + fz**2*bias_params.b4*kbins**2*(self.__norm**2*fz**2*48./143. + 48.*fz*bias_params.b1*self.__norm/77.+8.*bias_params.b1**2/35.)*(35./8.)*pk_mult[13]*self.__cosmo.h
                        )
            elif ell == 2:
                fz = self.__class.scale_independent_growth_factor_f(self.__redshift)
                pk_mult = self.__class.get_pk_mult(kbins * self.__cosmo.h, self.__redshift, len(kbins))

                return (self.__norm ** 2. * pk_mult[18]
                        + self.__norm ** 4. * (pk_mult[24])
                        + self.__norm ** 1. * bias_params.b1 * pk_mult[19]
                        + self.__norm ** 3. * bias_params.b1 * pk_mult[25]
                        + bias_params.b1 ** 2. * self.__norm ** 2. * pk_mult[26]
                        + bias_params.b1 * bias_params.b2 * self.__norm ** 2. * pk_mult[34]
                        + bias_params.b2 * self.__norm ** 3. * pk_mult[35]
                        + bias_params.b1 * bias_params.bG2 * self.__norm ** 2. * pk_mult[36]
                        + bias_params.bG2 * self.__norm ** 3. * pk_mult[37]
                        + 2. * (bias_params.css2 + 0. * bias_params.b4 * kbins ** 2. * self.__cosmo.h ** 2.) * self.__norm ** 2. * pk_mult[12] / self.__cosmo.h ** 2.
                        + (2. * bias_params.bG2 + 0.8 * bias_params.bGamma3) * self.__norm ** 3. * pk_mult[9]) * self.__cosmo.h ** 3.\
                       + fz ** 2. * bias_params.b4 * kbins ** 2. * ((self.__norm ** 2. * fz ** 2. * 70. + 165. * fz * bias_params.b1 * self.__norm + 99. * bias_params.b1 ** 2.) * 4. / 693.) * (35. / 8.) * pk_mult[13] * self.__cosmo.h \
                       + bias_params.a2 * (2. / 3.) * (kbins / 0.45)**2.

            elif ell == 0:
                fz = self.__class.scale_independent_growth_factor_f(self.__redshift)
                pk_mult = self.__class.get_pk_mult(kbins * self.__cosmo.h, self.__redshift, len(kbins))
                return (self.__norm ** 2. * pk_mult[15]
                        + self.__norm ** 4. * pk_mult[21]
                        + self.__norm ** 1. * bias_params.b1 * pk_mult[16]
                        + self.__norm ** 3. * bias_params.b1 * pk_mult[22]
                        + self.__norm ** 0. * bias_params.b1 ** 2. * pk_mult[17]
                        + self.__norm ** 2. * bias_params.b1 ** 2. * pk_mult[23]
                        + 0.25 * self.__norm ** 2. * bias_params.b2 ** 2. * pk_mult[1]
                        + bias_params.b1 * bias_params.b2 * self.__norm ** 2. * pk_mult[30]
                        + bias_params.b2 * self.__norm ** 3. * pk_mult[31]
                        + bias_params.b1 * bias_params.bG2 * self.__norm ** 2. * pk_mult[32]
                        + bias_params.bG2 * self.__norm ** 3. * pk_mult[33]
                        + bias_params.b2 * bias_params.bG2 * self.__norm ** 2. * pk_mult[4]
                        + bias_params.bG2 ** 2. * self.__norm ** 2. * pk_mult[5]
                        + 2. * bias_params.css0 * self.__norm ** 2. * pk_mult[11] / self.__cosmo.h ** 2.
                        + (2. * bias_params.bG2 + 0.8 * bias_params.bGamma3) * self.__norm ** 2. * (bias_params.b1 * pk_mult[7] + self.__norm * pk_mult[8])) * self.__cosmo.h ** 3. \
                        + bias_params.Pshot \
                        + bias_params.a0*(kbins/0.45)**2. \
                        + bias_params.a2 * (1. / 3.) * (kbins / 0.45)**2. \
                        + fz ** 2. * bias_params.b4 * kbins ** 2. * (self.__norm ** 2. * fz ** 2. / 9. + 2. * fz * bias_params.b1 * self.__norm / 7. + bias_params.b1 ** 2. / 5) * (35. / 8.) * pk_mult[13] * self.__cosmo.h

            else:
                raise ParameterValueError("The parameter ell has taken the unsupported value {}.".format(ell))

    def get_gaussian_covariance(self, kbins, bias_params, V_bin, Deltak=0.1, kNL=0.45, neff=3.3):
        if np.all(np.fabs(np.diff(np.log(kbins))-np.diff(np.log(kbins))[0])<1e-10):
            log_spaced=True
            deltak = np.log(kbins)[1]-np.log(kbins)[0]
        else:
            log_spaced=False
            assert np.all(np.fabs(np.diff(kbins)-np.diff(kbins)[0])<1e-10)
            deltak = kbins[1] - kbins[0]

        k_size=len(kbins)
        pk_mult = self.__class.get_pk_mult(kbins * self.__cosmo.h, self.__redshift, len(kbins))

        P0 = self.get_non_linear(0, kbins, bias_params)
        P2 = self.get_non_linear(2, kbins, bias_params)
        P4 = self.get_non_linear(4, kbins, bias_params)

        P0err = (self.__norm**2. * pk_mult[15] + self.__norm * bias_params.b1 * pk_mult[16] + bias_params.b1**2. * pk_mult[17]) * self.__cosmo.h**3. + bias_params.Pshot
        P2err = (self.__norm**2. * pk_mult[18] + self.__norm * bias_params.b1 * pk_mult[19]) * self.__cosmo.h**3.
        P4err = (self.__norm**2. * pk_mult[20]) * self.__cosmo.h**3.

        covmat = np.zeros((3*k_size, 3*k_size))
        if log_spaced:
            volume_denominator = V_bin * kbins**3. * 2. * np.pi * deltak
        else:
            volume_denominator = V_bin * kbins**2. * 2. * np.pi * deltak

        #cosmic variance errors
        cov00_diagonal = (2.*np.pi)**3.*(P0**2.+P2**2./5.+P4**2./9.)*10.**(-9.)/volume_denominator
        cov02_diagonal = (2.*np.pi)**3.*(2.*P0*P2 + 2.*P2**2./7.+4.*P2*P4/7.+100.*P4**2./693.)*10.**(-9.)/volume_denominator
        cov04_diagonal = (2.*np.pi)**3.*(2.*P0*P4 + 18.*P2**2./35.+40.*P2*P4/77.+162.*P4**2./1001.)*10.**(-9.)/volume_denominator

        cov22_diagonal = (2.*np.pi)**3.*(5.*P0**2.+20.*P0*P2/7.+ 20.*P0*P4/7.+ 15.*P2**2./7.+120.*P2*P4/77.+ P4**2.*8945./9009.)*10.**(-9.)/volume_denominator
        cov24_diagonal = (2.*np.pi)**3.*(36.*P0*P2/7.+ 200.*P0*P4/77.+ 108.*P2**2./77.+3578.*P2*P4/1001.+ P4**2.*900./1001.)*10.**(-9.)/volume_denominator

        cov44_diagonal = (2.*np.pi)**3.*(9.*P0**2.+360.*P0*P2/77.+ 2916.*P0*P4/1001.+ 16101.*P2**2./5005.+3240.*P2*P4/1001.+ P4**2.*42849./17017.)*10.**(-9.)/volume_denominator

        np.fill_diagonal(covmat[:k_size, :k_size], cov00_diagonal)
        np.fill_diagonal(covmat[k_size:2*k_size, :k_size], cov02_diagonal)
        np.fill_diagonal(covmat[:k_size, k_size:2*k_size], cov02_diagonal)
        np.fill_diagonal(covmat[2*k_size:3*k_size, :k_size], cov04_diagonal)
        np.fill_diagonal(covmat[:k_size, 2*k_size:3*k_size], cov04_diagonal)

        np.fill_diagonal(covmat[k_size:2*k_size, k_size:2*k_size], cov22_diagonal)
        np.fill_diagonal(covmat[k_size:2*k_size, 2*k_size:3*k_size], cov24_diagonal)
        np.fill_diagonal(covmat[2*k_size:3*k_size, k_size:2*k_size], cov24_diagonal)

        np.fill_diagonal(covmat[2*k_size:3*k_size, 2*k_size:3*k_size], cov44_diagonal)

        #theoretical errors
        k_meshX,k_meshY = np.meshgrid(kbins, kbins)
        k_diff = k_meshX-k_meshY

        Err0 = 1. * self.__cosmo.scale_independent_growth_factor(self.__redshift)**4. * P0err * (kbins / kNL)**neff
        Err2 = 1. * self.__cosmo.scale_independent_growth_factor(self.__redshift)**4. * P2err * (kbins / kNL)**neff
        Err4 = 1. * self.__cosmo.scale_independent_growth_factor(self.__redshift)**4. * P4 * (kbins / kNL)**neff

        err_list=[Err0, Err2, Err4]

        for i in range(0, 3):
            for j in range(0, 3):
                covmat[i*k_size:(i+1)*k_size, j*k_size:(j+1)*k_size] += np.outer(err_list[i], err_list[j])*np.exp(-1.*k_diff**2./2./Deltak**2.)

        return covmat

    def get_all_non_linear(self, kbins, bias_params):
        return self.get_non_linear(0, kbins, bias_params),self.get_non_linear(2, kbins, bias_params),self.get_non_linear(4, kbins, bias_params)


class NonLinearPowerReplace(NonLinearPower):
    def __init__(self, cosmo, linPower, redshift, Omfid=None, k_vals_h_invMpc=None, no_wiggle=False):

        super().__init__(cosmo, redshift, Omfid=Omfid, renormalize=True, no_wiggle=no_wiggle)
        self.__linPower = linPower

        if k_vals_h_invMpc is None:
            self.__k_vals = np.logspace(-5, 3, 10000)
        else:
            self.__k_vals = k_vals_h_invMpc

        pk_lin_vals = linPower(self.__k_vals, redshift)

        self.__temporary_power_spectrum_file = tempfile.NamedTemporaryFile()

        np.savetxt(self.get_power_spectrum_path(), np.vstack([self.__k_vals * cosmo.h, pk_lin_vals / cosmo.h**3 / self._NonLinearPower__norm**2.0]).T, delimiter='\t')

        class_non_linear_params = {'Input Pk': self.get_power_spectrum_path()}
        self._NonLinearPower__class.set(class_non_linear_params)

    def compute(self):
        try:
            super().compute()
        except ClassComputationError as ex:
            raise ClassComputationError("Non-linear power spectra computation failed.")
        finally:
            self.__temporary_power_spectrum_file.close()

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

    def replace_background_evolution(self, H, DA, D, f):
        replacement_params = {'replace background': 'YES',
                              'Hz_replace': H,
                              'DAz_replace': DA,
                              'Dz_replace': D,
                              'fz_replace': f}
        self.__class.set(replacement_params)
