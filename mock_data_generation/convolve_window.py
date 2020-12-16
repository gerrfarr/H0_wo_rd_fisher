import numpy as np
from numpy import pi, log, exp
from scipy import special,interpolate

class WindowConvolve(object):

    def __init__(self, window_function_path, k_vals):

        self.k=k_vals
        self.ksize=len(k_vals)

        self.Nmax = 128
        self.W0 = np.zeros((self.Nmax))
        self.W2 = np.zeros((self.Nmax))
        self.W4 = np.zeros((self.Nmax))
        datafile = open(window_function_path, 'r')
        for i in range(self.Nmax):
            line = datafile.readline()
            while line.find('#') != -1:
                line = datafile.readline()
            self.W0[i] = float(line.split()[0])
            self.W2[i] = float(line.split()[1])
            self.W4[i] = float(line.split()[2])
        datafile.close()

        # Precompute useful window function things
        kmax = 100.
        self.k0 = 1.e-4

        self.rmin = 0.01
        rmax = 1000.
        b = -1.1001
        bR = -2.001

        Delta = log(kmax / self.k0) / (self.Nmax - 1)
        Delta_r = log(rmax / self.rmin) / (self.Nmax - 1)
        i_arr = np.arange(self.Nmax)
        rtab = self.rmin * exp(Delta_r * i_arr)

        self.kbins3 = self.k0 * exp(Delta * i_arr)
        self.tmp_factor = exp(-1. * b * i_arr * Delta)
        self.tmp_factor2 = exp(-1. * bR * i_arr * Delta_r)

        jsNm = np.arange(-self.Nmax / 2, self.Nmax / 2 + 1, 1)
        self.etam = b + 2 * 1j * pi * (jsNm) / self.Nmax / Delta

        def J_func(r, nu):
            gam = special.gamma(2 + nu)
            r_pow = r**(-3. - 1. * nu)
            sin_nu = np.sin(pi * nu / 2.)
            J0 = -1. * sin_nu * r_pow * gam / (2. * pi**2.)
            J2 = r_pow * (3. + nu) * gam * sin_nu / (nu * 2. * pi**2.)
            return J0, J2

        self.J0_arr, self.J2_arr = J_func(rtab.reshape(-1, 1), self.etam.reshape(1, -1))

        self.etamR = bR + 2 * 1j * pi * (jsNm) / self.Nmax / Delta_r

        def Jk_func(k, nu):
            gam = special.gamma(2 + nu)
            k_pow = k**(-3. - 1. * nu)
            sin_nu = np.sin(pi * nu / 2.)
            J0k = -1. * k_pow * gam * sin_nu * (4. * pi)
            J2k = k_pow * (3. + nu) * gam * sin_nu * 4. * pi / nu
            return J0k, J2k

        self.J0k_arr, self.J2k_arr = Jk_func(self.kbins3.reshape(-1, 1), self.etamR.reshape(1, -1))

        # Compute window response matrix
        resp00 = np.zeros((self.ksize, self.Nmax))
        resp02 = np.zeros((self.ksize, self.Nmax))
        resp20 = np.zeros((self.ksize, self.Nmax))
        resp22 = np.zeros((self.ksize, self.Nmax))
        for i in range(self.Nmax):
            tmp_resp0 = self.window_response(0, i)
            tmp_resp2 = self.window_response(2, i)
            resp00[:, i] = tmp_resp0[0]
            resp20[:, i] = tmp_resp0[1]
            resp02[:, i] = tmp_resp2[0]
            resp22[:, i] = tmp_resp2[1]
        resp0 = np.hstack([resp00, resp02])
        resp2 = np.hstack([resp20, resp22])
        self.response_matrix = np.vstack([resp0, resp2])

    def window_response(self, l_i, k_index):
        Nmax = self.Nmax
        k0 = self.k0

        Pdiscrin0 = np.zeros(Nmax)
        Pdiscrin2 = np.zeros(Nmax)

        if l_i == 0:
            Pdiscrin0[k_index] = 1
        if l_i == 2:
            Pdiscrin2[k_index] = 1

        cm0 = np.fft.fft(Pdiscrin0) / Nmax
        cm2 = np.fft.fft(Pdiscrin2) / Nmax
        cmsym0 = np.zeros(Nmax + 1, dtype=np.complex_)
        cmsym2 = np.zeros(Nmax + 1, dtype=np.complex_)

        all_i = np.arange(Nmax + 1, dtype=np.int16)
        f = (all_i + 2 - Nmax / 2) < 1
        cmsym0[f] = k0**(-self.etam[f]) * np.conjugate(cm0[-all_i[f] + Nmax // 2])
        cmsym2[f] = k0**(-self.etam[f]) * np.conjugate(cm2[-all_i[f] + Nmax // 2])
        cmsym0[~f] = k0**(-self.etam[~f]) * cm0[all_i[~f] - Nmax // 2]
        cmsym2[~f] = k0**(-self.etam[~f]) * cm2[all_i[~f] - Nmax // 2]

        cmsym0[-1] = cmsym0[-1] / 2
        cmsym0[0] = cmsym0[0] / 2
        cmsym2[-1] = cmsym2[-1] / 2
        cmsym2[0] = cmsym2[0] / 2

        xi0 = np.real(cmsym0 * self.J0_arr).sum(axis=1)
        xi2 = np.real(cmsym2 * self.J2_arr).sum(axis=1)

        Xidiscrin0 = (xi0 * self.W0 + 0.2 * xi2 * self.W2) * self.tmp_factor2
        Xidiscrin2 = (xi0 * self.W2 + xi2 * (self.W0 + 2. * (self.W2 + self.W4) / 7.)) * self.tmp_factor2

        cmr0 = np.fft.fft(Xidiscrin0) / Nmax
        cmr2 = np.fft.fft(Xidiscrin2) / Nmax

        cmsymr0 = np.zeros(Nmax + 1, dtype=np.complex_)
        cmsymr2 = np.zeros(Nmax + 1, dtype=np.complex_)

        arr_i = np.arange(Nmax + 1, dtype=np.int16)
        f = (arr_i + 2 - Nmax / 2) < 1

        cmsymr0[f] = self.rmin**(-self.etamR[f]) * np.conjugate(cmr0[-arr_i[f] + Nmax // 2])
        cmsymr2[f] = self.rmin**(-self.etamR[f]) * np.conjugate(cmr2[-arr_i[f] + Nmax // 2])
        cmsymr0[~f] = self.rmin**(-self.etamR[~f]) * cmr0[arr_i[~f] - Nmax // 2]
        cmsymr2[~f] = self.rmin**(-self.etamR[~f]) * cmr2[arr_i[~f] - Nmax // 2]

        cmsymr0[-1] = cmsymr0[-1] / 2
        cmsymr0[0] = cmsymr0[0] / 2
        cmsymr2[-1] = cmsymr2[-1] / 2
        cmsymr2[0] = cmsymr2[0] / 2

        P0t = np.real(cmsymr0 * self.J0k_arr).sum(axis=1)
        P2t = np.real(cmsymr2 * self.J2k_arr).sum(axis=1)

        P0int = interpolate.InterpolatedUnivariateSpline(self.kbins3, P0t)(self.k)
        P2int = interpolate.InterpolatedUnivariateSpline(self.kbins3, P2t)(self.k)

        return P0int, P2int

    def __call__(self, preWindowData0, preWindowData2, h):
        factor = (exp(-1. * (self.kbins3 * h / 2.)**4.) * self.tmp_factor)
        Pdisc = np.hstack([preWindowData0 * factor, preWindowData2 * factor])
        return np.inner(self.response_matrix, Pdisc)

