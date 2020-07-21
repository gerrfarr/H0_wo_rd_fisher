"""
code to produce linear matter power spectrum using Eisenstein-Hu tranfer functions from transfer_functions.
heavily inspired by nbodykit

Jul 2020
"""
from .transfer_functions import *
A_s_PLANCK=2.0989e-9

class LinearPower(object):
    def __init__(self, cosmo, no_wiggle=False):
        self.cosmo=cosmo
        if no_wiggle:
            self._transfer = NoWiggleEisensteinHu(self.cosmo)
        else:
            self._transfer = EisensteinHu(self.cosmo)

        self._norm=1.0
        self._norm=(cosmo.sigma8 / self.sigma_r(8., 0.0))**2

    def __call__(self, k, z):
        Pk = k ** self.cosmo.n_s * self._transfer(k, z) ** 2
        return self._norm * Pk


    def sigma_r(self, r, z, kmin=1e-5, kmax=1e1):
        r"""
        The mass fluctuation within a sphere of radius ``r``, in
        units of :math:`h^{-1} Mpc` at ``redshift``.

        This returns :math:`\sigma`, where

        .. math::

            \sigma^2 = \int_0^\infty \frac{k^3 P(k,z)}{2\pi^2} W^2_T(kr) \frac{dk}{k},

        where :math:`W_T(x) = 3/x^3 (\mathrm{sin}x - x\mathrm{cos}x)` is
        a top-hat filter in Fourier space.

        The value of this function with ``r=8`` returns
        :attr:`sigma8`, within numerical precision.

        Parameters
        ----------
        r : float, array_like
            the scale to compute the mass fluctation over, in units of
            :math:`h^{-1} Mpc`
        kmin : float, optional
            the lower bound for the integral, in units of :math:`\mathrm{Mpc/h}`
        kmax : float, optional
            the upper bound for the integral, in units of :math:`\mathrm{Mpc/h}`
        """
        import mcfit
        from scipy.interpolate import InterpolatedUnivariateSpline as spline

        k = numpy.logspace(numpy.log10(kmin), numpy.log10(kmax), 1024)
        Pk = self(k, z)
        R, sigmasq = mcfit.TophatVar(k, lowring=True)(Pk, extrap=True)

        return spline(R, sigmasq)(r)**0.5
