"""
this file provides linear matter powerspectrum transfer functions.
This code is taken from nbodykit (https://github.com/bccp/nbodykit)

Jul 2020
"""

import numpy

class EisensteinHu():
    """
    The linear matter transfer function using the Eisenstein & Hu (1998)
    fitting formula with BAO wiggles.
    Parameters
    ----------
    cosmo : :class:`Cosmology`
        the cosmology instance
    References
    ----------
    Eisenstein & Hu, "Baryonic Features in the Matter Transfer Function", 1998
    """
    def __init__(self, cosmo):

        self.cosmo = cosmo

        self.Obh2 = cosmo.Omega0_b * cosmo.h ** 2
        self.Omh2 = (cosmo.Omega0_b + cosmo.Omega0_cdm) * cosmo.h ** 2
        self.f_baryon = self.Obh2 / self.Omh2
        self.theta_cmb = cosmo.T0_cmb / 2.7

        # redshift and wavenumber of equality
        self.z_eq = 2.5e4 * self.Omh2 * self.theta_cmb ** (-4) # this is 1 + z
        self.k_eq = 0.0746 * self.Omh2 * self.theta_cmb ** (-2) # units of 1/Mpc

        # sound horizon and k_silk
        self.z_drag_b1 = 0.313 * self.Omh2 ** -0.419 * (1 + 0.607 * self.Omh2 ** 0.674)
        self.z_drag_b2 = 0.238 * self.Omh2 ** 0.223
        self.z_drag    = 1291 * self.Omh2 ** 0.251 / (1. + 0.659 * self.Omh2 ** 0.828) * \
                           (1. + self.z_drag_b1 * self.Obh2 ** self.z_drag_b2)

        self.r_drag = 31.5 * self.Obh2 * self.theta_cmb ** -4 * (1000. / (1+self.z_drag))
        self.r_eq   = 31.5 * self.Obh2 * self.theta_cmb ** -4 * (1000. / self.z_eq)

        self.sound_horizon = cosmo.suppression_scaling * cosmo.sound_horizon_scaling * (2. / (3. * self.k_eq) * numpy.sqrt(6. / self.r_eq) * numpy.log((numpy.sqrt(1 + self.r_drag) + numpy.sqrt(self.r_drag + self.r_eq)) / (1 + numpy.sqrt(self.r_eq))))
        self.k_silk = 1.6 * self.Obh2 ** 0.52 * self.Omh2 ** 0.73 * (1 + (10.4*self.Omh2) ** -0.95)

        # alpha_c
        alpha_c_a1 = (46.9*self.Omh2) ** 0.670 * (1 + (32.1*self.Omh2) ** -0.532)
        alpha_c_a2 = (12.0*self.Omh2) ** 0.424 * (1 + (45.0*self.Omh2) ** -0.582)
        self.alpha_c = alpha_c_a1 ** -self.f_baryon * alpha_c_a2 ** (-self.f_baryon**3)

        # beta_c
        beta_c_b1 = 0.944 / (1 + (458*self.Omh2) ** -0.708)
        beta_c_b2 = 0.395 * self.Omh2 ** -0.0266
        self.beta_c = 1. / (1 + beta_c_b1 * ((1-self.f_baryon) ** beta_c_b2) - 1)

        y = self.z_eq / (1 + self.z_drag)
        alpha_b_G = y * (-6.*numpy.sqrt(1+y) + (2. + 3.*y) * numpy.log((numpy.sqrt(1+y)+1) / (numpy.sqrt(1+y)-1)))
        self.alpha_b = 2.07 *  self.k_eq * self.sound_horizon * (1+self.r_drag)**-0.75 * alpha_b_G

        self.beta_node = 8.41 * self.Omh2 ** 0.435
        self.beta_b    = 0.5 + self.f_baryon + (3. - 2.*self.f_baryon) * numpy.sqrt( (17.2*self.Omh2) ** 2 + 1 )

    def __call__(self, k, redshift, derivative=False, debug=False):
        r"""
        Return the Eisenstein-Hu transfer function with BAO wiggles.
        This is normalized to unity as :math:`k \rightarrow 0` at :math:`z=0`.
        The redshift scaling is provided by the
        :func:`Cosmology.scale_independent_growth_factor` function.
        Parameters
        ---------
        redshift : float
            redshift at which to evaluate transfer function
        k : float, array_like
            the wavenumbers in units of :math:`h \mathrm{Mpc}^{-1}`
        Returns
        -------
        Tk : float, array_like
            the transfer function evaluated at ``k``, normalized to unity on
            large scales
        """
        if numpy.isscalar(k) and k == 0.:
            return 1.0

        k = numpy.asarray(k)
        # only compute k > 0 modes
        valid = k > 0.

        k = k[valid] * self.cosmo.h # now in 1/Mpc
        q = k / (13.41*self.k_eq)
        ks = k*self.sound_horizon

        T_c_ln_beta   = numpy.log(numpy.e + 1.8*self.beta_c*q)
        T_c_ln_nobeta = numpy.log(numpy.e + 1.8*q);
        T_c_C_alpha   = 14.2 / self.alpha_c + 386. / (1 + 69.9 * q ** 1.08)
        T_c_C_noalpha = 14.2 + 386. / (1 + 69.9 * q ** 1.08)

        T_c_f = 1. / (1. + (ks/5.4) ** 4)
        f = lambda a, b : a / (a + b*q**2)
        T_c = T_c_f * f(T_c_ln_beta, T_c_C_noalpha) + (1-T_c_f) * f(T_c_ln_beta, T_c_C_alpha)

        dT_c_f = -4*T_c_f**2*(ks/5.4)**4
        dT_c = (f(T_c_ln_beta, T_c_C_noalpha) - f(T_c_ln_beta, T_c_C_alpha)) * dT_c_f

        s_tilde = self.sound_horizon * (1 + (self.beta_node/ks)**3) ** (-1./3.)
        ks_tilde = k*s_tilde
        dks_tilde = (ks_tilde**3*self.beta_node**3/ks**6+1)*ks_tilde

        T_b_T0 = f(T_c_ln_nobeta, T_c_C_noalpha)
        T_b_1 = T_b_T0 / (1 + (ks/5.2)**2 )
        T_b_2 = self.alpha_b / (1 + (self.beta_b/ks)**3 ) * numpy.exp(-(k/self.k_silk) ** 1.4)
        T_b = self.cosmo.peak_amp_scaling*numpy.sin(ks_tilde / self.cosmo.suppression_scaling)/ks_tilde * (T_b_1 + T_b_2)

        dT_b_1=-T_b_1**2/T_b_T0*2*(ks/5.2)**2
        dT_b_2=T_b_2*( 1+3*(self.beta_b/ks)**3/(1+(self.beta_b/ks)**3) )
        dT_b = (ks_tilde*numpy.cos(ks_tilde) - numpy.sin(ks_tilde))/ks_tilde**2*dks_tilde*(T_b_1+T_b_2) + self.cosmo.peak_amp_scaling*numpy.sin(ks_tilde)/ks_tilde*(dT_b_1 + dT_b_2)

        T = numpy.ones(valid.shape)
        T[valid] = self.f_baryon*T_b + (1-self.f_baryon)*T_c;
        dT=numpy.zeros(valid.shape)
        dT[valid]=self.f_baryon*dT_b + (1-self.f_baryon)*dT_c;
        if debug:
            return T_c_f, dT_c_f, T_c, dT_c, ks_tilde, dks_tilde, T_b_1, dT_b_1, T_b_2, dT_b_2, T_b, dT_b

        if derivative:
            return T * self.cosmo.scale_independent_growth_factor(redshift),dT * self.cosmo.scale_independent_growth_factor(redshift)
        else:
            return T * self.cosmo.scale_independent_growth_factor(redshift)

class NoWiggleEisensteinHu(object):
    """
    Linear power spectrum using the Eisenstein & Hu (1998) fitting formula
    without BAO wiggles.
    Parameters
    ----------
    cosmo : :class:`Cosmology`
        the cosmology instance
    References
    ----------
    Eisenstein & Hu, "Baryonic Features in the Matter Transfer Function", 1998
    """
    def __init__(self, cosmo):
        self.cosmo = cosmo

        self.Obh2      = cosmo.Omega0_b * cosmo.h ** 2
        self.Omh2      = cosmo.Omega0_m * cosmo.h ** 2
        self.f_baryon  = cosmo.Omega0_b / cosmo.Omega0_m
        self.theta_cmb = cosmo.T0_cmb / 2.7

        # wavenumber of equality
        self.k_eq = 0.0746 * self.Omh2 * self.theta_cmb ** (-2) # units of 1/Mpc

        self.sound_horizon = cosmo.suppression_scaling * cosmo.sound_horizon_scaling * cosmo.h * 44.5 * numpy.log(9.83 / self.Omh2) / \
                             numpy.sqrt(1 + 10 * self.Obh2** 0.75) # in Mpc/h
        self.alpha_gamma = 1 - 0.328 * numpy.log(431*self.Omh2) * self.f_baryon + \
                            0.38* numpy.log(22.3*self.Omh2) * self.f_baryon ** 2


    def __call__(self, k, redshift, derivative=False, debug=False):
        r"""
        Return the Eisenstein-Hu transfer function without BAO wiggles.
        This is normalized to unity as :math:`k \rightarrow 0` at :math:`z=0`.
        The redshift scaling is provided by the
        :func:`~Cosmology.scale_independent_growth_factor` function.
        Parameters
        ---------
        redshift : float
            redshift at which to evaluate transfer function
        k : float, array_like
            the wavenumbers in units of :math:`h \mathrm{Mpc}^{-1}`
        Returns
        -------
        Tk : float, array_like
            the transfer function evaluated at ``k``, normalized to unity on
            large scales
        """
        if numpy.isscalar(k) and k == 0.:
            return 1.0

        # only compute k > 0 modes
        k = numpy.asarray(k)
        valid = k > 0.

        k = k[valid] * self.cosmo.h # in 1/Mpc now
        ks = k * self.sound_horizon / self.cosmo.h
        q = k / (13.41*self.k_eq)

        gamma_eff = self.Omh2 * (self.alpha_gamma + (1 - self.alpha_gamma) / (1 + (0.43*ks) ** 4))
        dgamma_eff_dks = -4 * 0.43**4*ks**3*(1-self.alpha_gamma)/(1 + (0.43*ks) ** 4)**2*self.Omh2

        q_eff = q * self.Omh2 / gamma_eff
        dq_eff_dgamma_eff = -q_eff/gamma_eff

        L0 = numpy.log(2*numpy.e + 1.8 * q_eff)
        dL0_dq_eff = 1.8/(2*numpy.e+1.8*q_eff)

        C0 = 14.2 + 731.0 / (1 + 62.5 * q_eff)
        dC0_dq_eff = - 731.0*62.5/(1+62.5*q_eff)**2

        T = numpy.ones(valid.shape)
        T[valid] = L0 / (L0 + C0 * q_eff**2)
        dT = numpy.zeros(valid.shape)
        dT[valid] = (dL0_dq_eff*T/L0 - T**2/L0*(2*q_eff*C0 + q_eff**2* dC0_dq_eff + dL0_dq_eff))*dq_eff_dgamma_eff*dgamma_eff_dks*ks
        if debug:
            return ks, gamma_eff, dgamma_eff_dks, q_eff, dq_eff_dgamma_eff, L0, dL0_dq_eff, C0, dC0_dq_eff
        if derivative:
            return T * self.cosmo.scale_independent_growth_factor(redshift), dT*self.cosmo.scale_independent_growth_factor(redshift)
        else:
            return T * self.cosmo.scale_independent_growth_factor(redshift)