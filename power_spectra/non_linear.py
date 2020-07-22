import tempfile
import numpy as np
import _warnings as warnings
from .linear_power import LinearPower
from .cosmology import Cosmology
from ..custom_exceptions import ClassComputationError, OrderOfOperationsError, ParameterValueError, TemporaryFileClosedWarning


class BiasParams(object):
    def __init__(self, b1=2.0, b2=0.0, bG2=0.0, css0=0.0, css2=0.0, b4=500.0, Pshot=0.0, bGamma3=0.0):
        self.b1 = b1
        self.b2 = b2
        self.bG2 = bG2
        self.css0 = css0
        self.css2 = css2
        self.b4 = b4
        self.Pshot = Pshot
        self.bGamma3 = bGamma3


class NonLinearPower(object):
    def __init__(self, cosmo, linPower, redshift, k_vals_h_invMpc=None):
        """
        :type cosmo: Cosmology
        :type linPower: LinearPower
        :type redshift: float
        :type bias_params: BiasParams
        """

        self.__cosmo = cosmo
        self.__linPower = linPower
        self.__redshift = redshift
        self.__bias_params = None
        self.__kbins = None
        self.__computed = False
        self.__bias_set = False

        if k_vals_h_invMpc is None:
            k_vals_h_invMpc = np.logspace(-5, 3, 10000)

        pk_lin_vals = linPower(k_vals_h_invMpc, redshift)

        self.__temporary_power_spectrum_file = tempfile.NamedTemporaryFile()

        np.savetxt(self.get_power_spectrum_path(), np.vstack([k_vals_h_invMpc * cosmo.h, pk_lin_vals / cosmo.h ** 3]).T, delimiter='\t')

        self.__class_params = cosmo.class_params()
        self.__class = cosmo.get_class()

        class_non_linear_params = {'z_pk': redshift,
                                   'non linear': ' PT ',
                                   'IR resummation': ' Yes ',
                                   'Bias tracers': ' Yes ',
                                   'RSD': ' Yes ',
                                   'AP': 'Yes',
                                   'FFTLog mode': 'FAST',
                                   'Input Pk': self.get_power_spectrum_path(),
                                   }

        self.__class.set(class_non_linear_params)

    @property
    def computed(self):
        return self.__computed and self.__bias_set

    def set_bias_and_bins(self, kbins, bias_params):
        self.__bias_params = bias_params
        self.__kbins = kbins
        self.__bias_set = True

    def compute(self):
        try:
            self.__class.compute()
        except Exception as ex:
            print("Got the following error while computing non-linear power specta:")
            print(str(ex))
            raise ClassComputationError("Non-linear power spectra computation failed.")
        self.__computed = True
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

    def get_non_linear(self, ell):
        if not self.__computed or not self.__bias_set:
            raise OrderOfOperationsError("Non-linear power spectra can not be obtained. They have not been computed or the bias parameters have not been set.")
        else:
            if ell == 0:
                fz = self.__class.scale_independent_growth_factor_f(self.__redshift)
                pk_mult = self.__class.get_pk_mult(self.__kbins * self.__cosmo.h, self.__redshift, len(self.__kbins))

                return (self.__cosmo.norm ** 2. * pk_mult[18] \
                        + self.__cosmo.norm ** 4. * (pk_mult[24]) \
                        + self.__cosmo.norm ** 1. * self.__bias_params.b1 * pk_mult[19] + self.__cosmo.norm ** 3. * self.__bias_params.b1 * (pk_mult[25]) \
                        + self.__bias_params.b1 ** 2. * self.__cosmo.norm ** 2. * pk_mult[26] \
                        + self.__bias_params.b1 * self.__bias_params.b2 * self.__cosmo.norm ** 2. * pk_mult[34] \
                        + self.__bias_params.b2 * self.__cosmo.norm ** 3. * pk_mult[35] \
                        + self.__bias_params.b1 * self.__bias_params.bG2 * self.__cosmo.norm ** 2. * pk_mult[36] \
                        + self.__bias_params.bG2 * self.__cosmo.norm ** 3. * pk_mult[37] \
                        + 2. * (self.__bias_params.css2 + 0. * self.__bias_params.b4 * self.__kbins ** 2. * self.__cosmo.h ** 2.) * self.__cosmo.norm ** 2. * pk_mult[12] / self.__cosmo.h ** 2. \
                        + (2. * self.__bias_params.bG2 + 0.8 * self.__bias_params.bGamma3) * self.__cosmo.norm ** 3. * pk_mult[9]) * self.__cosmo.h ** 3. \
                       + fz ** 2. * self.__bias_params.b4 * self.__kbins ** 2. * ((self.__cosmo.norm ** 2. * fz ** 2. * 70. + 165. * fz * self.__bias_params.b1 * self.__cosmo.norm + 99. * self.__bias_params.b1 ** 2.) * 4. / 693.) * (35. / 8.) * pk_mult[13] * self.__cosmo.h

            elif ell == 2:
                fz = self.__class.scale_independent_growth_factor_f(self.__redshift)
                pk_mult = self.__class.get_pk_mult(self.__kbins * self.__cosmo.h, self.__redshift, len(self.__kbins))
                return (self.__cosmo.norm ** 2. * pk_mult[15] \
                        + self.__cosmo.norm ** 4. * pk_mult[21] \
                        + self.__cosmo.norm ** 1. * self.__bias_params.b1 * pk_mult[16] \
                        + self.__cosmo.norm ** 3. * self.__bias_params.b1 * pk_mult[22] \
                        + self.__cosmo.norm ** 0. * self.__bias_params.b1 ** 2. * pk_mult[17] \
                        + self.__cosmo.norm ** 2. * self.__bias_params.b1 ** 2. * pk_mult[23] \
                        + 0.25 * self.__cosmo.norm ** 2. * self.__bias_params.b2 ** 2. * pk_mult[1] \
                        + self.__bias_params.b1 * self.__bias_params.b2 * self.__cosmo.norm ** 2. * pk_mult[30] \
                        + self.__bias_params.b2 * self.__cosmo.norm ** 3. * pk_mult[31] \
                        + self.__bias_params.b1 * self.__bias_params.bG2 * self.__cosmo.norm ** 2. * pk_mult[32] \
                        + self.__bias_params.bG2 * self.__cosmo.norm ** 3. * pk_mult[33] \
                        + self.__bias_params.b2 * self.__bias_params.bG2 * self.__cosmo.norm ** 2. * pk_mult[4] \
                        + self.__bias_params.bG2 ** 2. * self.__cosmo.norm ** 2. * pk_mult[5] \
                        + 2. * self.__bias_params.css0 * self.__cosmo.norm ** 2. * pk_mult[11] / self.__cosmo.h ** 2. \
                        + (2. * self.__bias_params.bG2 + 0.8 * self.__bias_params.bGamma3) * self.__cosmo.norm ** 2. * (self.__bias_params.b1 * pk_mult[7] + self.__cosmo.norm * pk_mult[8])) * self.__cosmo.h ** 3. \
                       + self.__bias_params.Pshot \
                       + fz ** 2. * self.__bias_params.b4 * self.__kbins ** 2. * (self.__cosmo.norm ** 2. * fz ** 2. / 9. + 2. * fz * self.__bias_params.b1 * self.__cosmo.norm / 7. + self.__bias_params.b1 ** 2. / 5) * (35. / 8.) * pk_mult[13] * self.__cosmo.h

            else:
                raise ParameterValueError("The parameter ell has taken the unsupported value {}.".format(ell))
