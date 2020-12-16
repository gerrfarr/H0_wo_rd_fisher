"""
code to produce linear matter power spectrum using Eisenstein-Hu tranfer functions from transfer_functions.
heavily inspired by nbodykit

Jul 2020
Gerrit Farren
"""
from .transfer_functions import *
from .cosmology import Cosmology

class LinearPower(object):


    def __init__(self, cosmo, no_wiggle=False, sigma8_norm=None):
        """

        Parameters
        ----------
        cosmo : Cosmology
            Instance of cosmology
        no_wiggle : bool, optional
            Whether to use :class:`NoWiggleEisensteinHu` or :class:`EisensteinHu` (default)

        """
        self.cosmo=cosmo
        if no_wiggle:
            self._transfer = NoWiggleEisensteinHu(self.cosmo)
        else:
            self._transfer = EisensteinHu(self.cosmo)

        if sigma8_norm is None:
            sigma8_norm = 0.8277107664698417
        self._norm=1.0
        self._norm=cosmo.norm*(sigma8_norm / self.sigma_r(8., 0.0))**2

    def __call__(self, k, z):
        r"""
        Parameters
        ----------
        k : array_like
            k values in :math:`\mathrm{Mpc/h}` to compute power spectrum at
        z : float
            redshift

        Returns
        __________
        array_like
            linear power spectrum at redshift `z` and wavenumber(s) `k`
        """
        Pk = k ** self.cosmo.n_s * self._transfer(k, z) ** 2
        return self._norm * Pk

    def derivative(self, k, z):#derivative with respect to sound horizon
        T, dT = self._transfer(k, z, derivative=True)
        dPk = 2*k ** self.cosmo.n_s * T * dT
        return self._norm*dPk

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
            the scale to compute the mass fluctation over, in units of :math:`h^{-1} Mpc`
        z: float
            the redshift to compute the mass fluctuation at
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
