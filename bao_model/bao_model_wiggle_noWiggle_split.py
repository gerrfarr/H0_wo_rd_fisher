import numpy as np
import os
import os
import numpy as np
import warnings
from scipy.interpolate import interp1d,InterpolatedUnivariateSpline
from scipy.special import gamma
from scipy.integrate import simps
from scipy.special.orthogonal import p_roots
from classy import Class
from scipy.special import eval_legendre as legendre

class BAOPowerModel(object):
    # Set redshifts
    z = [0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2.]

    # other input parameters
    data_directory = "/Users/gerrit/SynologyDrive/Cambridge/H0_project/data/EUCLID_mock_spectra/"

    theory_pk_file = ["LCDM_spectra/euclid_mock_lcdm_z1.dat", "LCDM_spectra/euclid_mock_lcdm_z2.dat", "LCDM_spectra/euclid_mock_lcdm_z3.dat", "LCDM_spectra/euclid_mock_lcdm_z4.dat", "LCDM_spectra/euclid_mock_lcdm_z5.dat", "LCDM_spectra/euclid_mock_lcdm_z6.dat", "LCDM_spectra/euclid_mock_lcdm_z7.dat", "LCDM_spectra/euclid_mock_lcdm_z8.dat"]

    theory_pk_file_nw = ["LCDM_nw_spectra_new/euclid_mock_lcdm_z1.dat", "LCDM_nw_spectra_new/euclid_mock_lcdm_z2.dat", "LCDM_nw_spectra_new/euclid_mock_lcdm_z3.dat", "LCDM_nw_spectra_new/euclid_mock_lcdm_z4.dat", "LCDM_nw_spectra_new/euclid_mock_lcdm_z5.dat", "LCDM_nw_spectra_new/euclid_mock_lcdm_z6.dat", "LCDM_nw_spectra_new/euclid_mock_lcdm_z7.dat", "LCDM_nw_spectra_new/euclid_mock_lcdm_z8.dat"]

    cov_file = ["LCDM_covmats/euclid_mock_covmat_z1.dat", "LCDM_covmats/euclid_mock_covmat_z2.dat", "LCDM_covmats/euclid_mock_covmat_z3.dat", "LCDM_covmats/euclid_mock_covmat_z4.dat", "LCDM_covmats/euclid_mock_covmat_z5.dat", "LCDM_covmats/euclid_mock_covmat_z6.dat", "LCDM_covmats/euclid_mock_covmat_z7.dat", "LCDM_covmats/euclid_mock_covmat_z8.dat"]

    cov_file_nw = ["LCDM_nw_covmats_new/euclid_mock_covmat_z1.dat", "LCDM_nw_covmats_new/euclid_mock_covmat_z2.dat", "LCDM_nw_covmats_new/euclid_mock_covmat_z3.dat", "LCDM_nw_covmats_new/euclid_mock_covmat_z4.dat", "LCDM_nw_covmats_new/euclid_mock_covmat_z5.dat", "LCDM_nw_covmats_new/euclid_mock_covmat_z6.dat", "LCDM_nw_covmats_new/euclid_mock_covmat_z7.dat", "LCDM_nw_covmats_new/euclid_mock_covmat_z8.dat"]

    theory_fid_params = {'h': 0.6821,
                            'omega_b': 0.02253,
                            'omega_cdm': 0.1177,
                            'A_s': 2.216e-9,
                            'n_s': 0.9686,
                            'tau_reio': 0.085,
                            'm_ncdm': 0.06,
                            'N_ncdm': 1,
                            'N_ur': 2.0328}

    use_quadrupole = True
    use_hexadecapole = True

    Delta_k = 0.1



    def __init__(self, n_gauss=30, inflate_error=False):
        self.z = np.asarray(self.z)
        self.n_bin = np.shape(self.z)[0]
        self.inflate_error = inflate_error

        [self.gauss_mu, self.gauss_w] = p_roots(n_gauss)

        self.cosmo_fid = Class()
        self.cosmo_fid.set(self.theory_fid_params)
        self.cosmo_fid.compute()

        self.k_vals, self.Pk_theory_interp = self.interpolate_theory_spectrum(self.theory_pk_file)
        dump, self.Pk_theory_nw_interp = self.interpolate_theory_spectrum(self.theory_pk_file_nw)

        self.wiggle_only_interp = lambda k, mu: self.Pk_theory_interp(k, mu)-self.Pk_theory_nw_interp(k, mu)


        # Load in covariance matrices
        self.all_cov = np.zeros((self.n_bin, (1 + self.use_quadrupole + self.use_hexadecapole) * len(self.k_vals), (1 + self.use_quadrupole + self.use_hexadecapole) * len(self.k_vals)))
        for index_z in range(self.n_bin):
            this_cov = np.loadtxt(os.path.join(self.data_directory, self.cov_file_nw[index_z]))

            if self.use_quadrupole and self.use_hexadecapole:
                if len(this_cov) == len(self.k_vals) * 3:
                    pass
                else:
                    raise Exception('Need correct size covariance for monopole+quadrupole+hexadecapole analysis')
            elif self.use_quadrupole and self.use_hexadecapole:
                if len(this_cov) == len(self.k_vals) * 2:
                    pass
                elif len(this_cov) == len(self.k_vals) * 3:
                    this_cov = this_cov[:2 * len(self.k_vals), :2 * len(self.k_vals)]
                else:
                    raise Exception('Need correct size covariance for monopole+quadrupole analysis')
            else:
                if len(this_cov) == len(self.k_vals):
                    pass
                elif len(this_cov) == len(self.k_vals) * 2 or len(this_cov) == len(self.k_vals) * 3:
                    this_cov = this_cov[:len(self.k_vals), :len(self.k_vals)]
                else:
                    raise Exception('Need correct size covariance for monopole-only analysis')

            self.all_cov[index_z] = this_cov

        self.full_cov = np.zeros(self.all_cov.shape)
        ## Compute theoretical error envelope and combine with usual covariance
        P0, P2, P4 = self.pk_model(self.k_vals, np.ones(self.z.shape), np.ones(self.z.shape))
        for index_z in range(self.n_bin):
            # Compute the linear (fiducial) power spectrum from CLASS
            envelope_power0 = P0[index_z] * 2  # extra factor to ensure we don't underestimate error
            envelope_power2 = envelope_power0 * np.sqrt(5.)  # rescale by sqrt{2ell+1}
            envelope_power4 = envelope_power0 * np.sqrt(9.)  # rescale by sqrt{2ell+1}

            # Define model power
            if self.inflate_error:
                envelope_power0 *= 5.
                envelope_power2 *= 5.
                envelope_power4 *= 5.

            ## COMPUTE THEORETICAL ERROR COVARIANCE
            # Define coupling matrix
            k_mats = np.meshgrid(self.k_vals, self.k_vals)
            diff_k = k_mats[0] - k_mats[1]
            rho_submatrix = np.exp(-diff_k**2. / (2. * self.Delta_k**2.))

            if self.use_quadrupole and self.use_hexadecapole:
                # Assume uncorrelated multipoles here
                zero_matrix = np.zeros_like(rho_submatrix)
                rho_matrix = np.hstack([np.vstack([rho_submatrix, zero_matrix, zero_matrix]),
                                        np.vstack([zero_matrix, rho_submatrix, zero_matrix]),
                                        np.vstack([zero_matrix, zero_matrix, rho_submatrix])])
            elif self.use_quadrupole:
                zero_matrix = np.zeros_like(rho_submatrix)
                rho_matrix = np.hstack([np.vstack([rho_submatrix, zero_matrix]),
                                        np.vstack([zero_matrix, rho_submatrix])])
            else:
                rho_matrix = rho_submatrix

            # Define error envelope from Baldauf'16

            E_vector0 = np.power(self.k_vals / 0.31, 1.8) * envelope_power0
            if self.use_quadrupole and self.use_hexadecapole:
                E_vector2 = np.power(self.k_vals / 0.31, 1.8) * envelope_power2
                E_vector4 = np.power(self.k_vals / 0.31, 1.8) * envelope_power4
                stacked_E = np.concatenate([E_vector0, E_vector2, E_vector4])
            elif self.use_quadrupole:
                E_vector2 = np.power(self.k_vals / 0.31, 1.8) * envelope_power2
                stacked_E = np.concatenate([E_vector0, E_vector2])
            else:
                stacked_E = E_vector0

            E_mat = np.diag(stacked_E)
            cov_theoretical_error = np.matmul(E_mat, np.matmul(rho_matrix, E_mat))

            # Compute precision matrix
            self.full_cov[index_z] = cov_theoretical_error + self.all_cov[index_z]

    def interpolate_theory_spectrum(self, files):
        # load theory power spectra
        for index_z in range(self.n_bin):
            data = np.loadtxt(os.path.join(self.data_directory, files[index_z]))

            if index_z == 0:
                k_vals_theory = data[:, 0]
                Pk0_theory = np.zeros((self.n_bin, len(data[:, 0])))
                Pk2_theory = np.zeros((self.n_bin, len(data[:, 0])))
                Pk4_theory = np.zeros((self.n_bin, len(data[:, 0])))

            Pk0_theory[index_z] = data[:, 1]
            if self.use_quadrupole:
                Pk2_theory[index_z] = data[:, 2]
            if self.use_hexadecapole:
                Pk4_theory[index_z] = data[:, 3]

        Pk0_theory_interp = interp1d(k_vals_theory, Pk0_theory, fill_value="extrapolate")
        if self.use_quadrupole and self.use_hexadecapole:
            Pk2_theory_interp = interp1d(k_vals_theory, Pk2_theory, fill_value="extrapolate")
            Pk4_theory_interp = interp1d(k_vals_theory, Pk4_theory, fill_value="extrapolate")

            Pk_theory_interp = lambda k, mu: Pk0_theory_interp(k) * legendre(0, mu) + Pk2_theory_interp(k) * legendre(2, mu) + Pk4_theory_interp(k) * legendre(4, mu)
        elif self.use_quadrupole:
            Pk2_theory_interp = interp1d(k_vals_theory, Pk2_theory, fill_value="extrapolate")
            Pk_theory_interp = lambda k, mu: Pk0_theory_interp(k) * legendre(0, mu) + Pk2_theory_interp(k) * legendre(2, mu)
        else:
            Pk_theory_interp = lambda k, mu: Pk0_theory_interp(k) * legendre(0, mu)

        return k_vals_theory, Pk_theory_interp


    def pk_model(self, k_vals, alpha_perp, alpha_par, norm=1.0):
        ## must be in h/Mpc units
        ## Returns linear theory prediction for monopole and quadrupole

        muGrid, kGrid = np.meshgrid(self.gauss_mu, k_vals)
        wGrid = np.einsum('i,j->ij', np.ones(k_vals.shape), self.gauss_w)[np.newaxis, :, :]

        ## Compute AP-rescaled parameters
        F = alpha_par / alpha_perp

        k1 = kGrid[np.newaxis,:,:] / alpha_perp[:, np.newaxis, np.newaxis] * np.sqrt(1. + np.power(muGrid[np.newaxis,:,:], 2.) * (np.power(F[:, np.newaxis, np.newaxis], -2.) - 1.))
        mu1 = muGrid[np.newaxis,:,:] / (F[:, np.newaxis, np.newaxis] * np.sqrt(1. + np.power(muGrid[np.newaxis,:,:], 2.) * (np.power(F[:, np.newaxis, np.newaxis], -2.) - 1.)))

        P_k_mu = self.Pk_theory_nw_interp(kGrid, muGrid)+self.wiggle_only_interp(k1, mu1).diagonal(axis1=0, axis2=1).transpose([2, 0, 1])

        # Use Gaussian quadrature for fast integral evaluation
        P0_est = np.sum(P_k_mu*wGrid, axis=-1) / 2.

        if self.use_quadrupole:
            P2_est = np.sum(P_k_mu*legendre(2, muGrid)[np.newaxis, :, :]*wGrid, axis=-1)*5. / 2.
        else:
            P2_est = 0.

        if self.use_hexadecapole:
            P4_est = np.sum(P_k_mu*legendre(4, muGrid)[np.newaxis, :, :]*wGrid, axis=-1)*9. / 2.
        else:
            P4_est = 0.

        return norm*P0_est, norm*P2_est, norm*P4_est

    def get_AP_params(self, cosmo):
        alpha_par = self.cosmo_fid.z_of_r(self.z)[1] * self.cosmo_fid.rs_drag() / (cosmo.z_of_r(self.z)[1] * cosmo.rs_drag())
        alpha_perp = cosmo.z_of_r(self.z)[0] * self.cosmo_fid.rs_drag() / (self.cosmo_fid.z_of_r(self.z)[0] * cosmo.rs_drag())

        return alpha_par, alpha_perp

    def eval(self, k_vals, cosmo, norm=1.0):
        alpha_par, alpha_perp = self.get_AP_params(cosmo)
        return self.pk_model(k_vals, alpha_par, alpha_perp, norm=norm)

    def eval_fid(self, k_vals, norm=1.0):
        alpha_par, alpha_perp = self.get_AP_params(self.cosmo_fid)
        return self.pk_model(k_vals, alpha_par, alpha_perp, norm=norm)

    def get_covariance(self):
        return self.full_cov