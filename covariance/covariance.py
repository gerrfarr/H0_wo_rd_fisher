import numpy as np


from ..power_spectra.non_linear import NonLinearPower
from ..power_spectra.cosmology import Cosmology
from ..power_spectra.non_linear import BiasParams

class CovarianceMatrix(object):
    def __init__(self, cosmology, non_linear_power, kbins, bias_params, V_eff=2.42e9):
        """

        Parameters
        ----------
        cosmology : Cosmology
        non_linear_power : NonLinearPower
        bias_params : BiasParams
        V_eff : float

        """
        self.__cosmo = cosmology
        self.__non_linear = non_linear_power
        self.__kbins = kbins
        self.__bias_params = bias_params
        self.__V_eff = V_eff

        self.__P_0_spectrum = self.__non_linear.get_non_linear(0, self.__kbins, self.__bias_params)
        self.__P_2_spectrum = self.__non_linear.get_non_linear(2, self.__kbins, self.__bias_params)
        self.__bin_volumes = self.bin_volume(kbins, kbins[1]-kbins[0])

        self.__covariance_matrix = np.zeros((2 * len(self.__kbins), 2 * len(self.__kbins)))
        self.__covariance_matrix[0:len(self.__kbins),0:len(self.__kbins)] = self.compute_00()
        self.__covariance_matrix[len(self.__kbins):2*len(self.__kbins), len(self.__kbins):2*len(self.__kbins)] = self.compute_22()
        self.__covariance_matrix[len(self.__kbins):2*len(self.__kbins), 0:len(self.__kbins)] = self.compute_02()
        self.__covariance_matrix[0:len(self.__kbins), len(self.__kbins):2*len(self.__kbins)] = self.compute_02()

    def __call__(self):
        return self.__covariance_matrix

    def compute_00(self):
        matrix = np.zeros((len(self.__kbins), len(self.__kbins)))
        values = 2 * (2 * np.pi)**3. / self.__V_eff / self.__bin_volumes * (self.__P_0_spectrum**2. + self.__P_2_spectrum**2. / 5)
        np.fill_diagonal(matrix, values)

        return matrix


    def compute_02(self):
        matrix = np.zeros((len(self.__kbins), len(self.__kbins)))
        values = 2 * (2 * np.pi)**3. / self.__V_eff / self.__bin_volumes * (2. / 7. * self.__P_2_spectrum * (7. * self.__P_0_spectrum + self.__P_2_spectrum))
        np.fill_diagonal(matrix, values)

        return matrix

    def compute_22(self):
        matrix = np.zeros((len(self.__kbins), len(self.__kbins)))
        values = 2 * (2 * np.pi)**3. / self.__V_eff / self.__bin_volumes * (5. / 7. * (7. * self.__P_0_spectrum**2. + 4. * self.__P_0_spectrum * self.__P_2_spectrum + 3. * self.__P_2_spectrum**2.))
        np.fill_diagonal(matrix, values)

        return matrix

    def bin_volume(self, k, bin_width):
        return 4*np.pi*k**2*bin_width
