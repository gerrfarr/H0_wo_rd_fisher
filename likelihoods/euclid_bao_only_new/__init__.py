from montepython.likelihood_class import Likelihood
import os
import numpy as np
from scipy.special.orthogonal import p_roots
from scipy.special import eval_legendre as legendre

from classy_pt import Class

class euclid_bao_only_new(Likelihood):


    def __init__(self, path,data,command_line):

        Likelihood.__init__(self, path, data, command_line)

        self.z = np.asarray(self.z)
        self.n_bin = np.shape(self.z)[0]

        self.cosmo_fid = Class()
        self.cosmo_fid.set({'h': 0.6821,
                            'omega_b': 0.02253,
                            'omega_cdm': 0.1177,
                            'A_s': 2.216e-9,
                            'n_s': 0.9686,
                            'tau_reio': 0.085,
                            'm_ncdm': 0.06,
                            'N_ncdm': 1,
                            'N_ur': 2.0328,
                            'output': 'mPk',
                            'output format': 'FAST',
                            'FFTLog mode': 'FAST',
                            'P_k_max_h/Mpc': 100,
                            'non linear': 'PT',
                            'IR resummation': 'YES',
                            'Bias tracers': 'YES',
                            'RSD': 'YES',
                            'AP': 'YES',
                            'Omfid': 0.3027857053007527,
                            'z_pk': '0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0',
                            'z_max_pk': 10.})
        self.cosmo_fid.compute()

        self.b1fid = 0.9 + 0.4 * self.z
        self.bG2fid = -2. / 7. * (self.b1fid - 1.)
        self.b2fid = -0.704172 - 0.207993 * self.z + 0.183023 * self.z**2. - 0.00771288 * self.z**3. + 4. / 3. * self.bG2fid
        self.bGamma3fid = 23. / 42. * (self.b1fid - 1.)

        self.css0fid = np.array([1.028409076422109, 0.8504750555126785, 0.710998446651058, 0.6009290940289844, 0.5132410445958041, 0.442639546298396, 0.38517762925674237, 0.33791511114016776])
        self.css2fid = np.array([28.145932617868244, 23.276159414031202, 19.45890485571317, 16.44648046816168, 14.04659700999043, 12.114345477640311, 10.54170353755295, 9.248203041730907])
        self.css4fid = np.array([-1.2990430439016112, -1.0742842806475938, -0.8981033010329155, -0.7590683292997699, -0.6483044773841736, -0.5591236374295528, -0.4865401632716746, -0.4268401403875803])
        # NB: we don't include b4 term here
        self.b4fid = np.zeros(self.n_bin, 'float64')
        self.Pshotfid = np.array([266.72015936025565, 493.3458318663388, 871.8004574255534, 1483.8895054255129, 2643.3492511153418, 4942.814802630568, 8865.13543105212, 15323.468044314039])

        self.poly_priors = np.array([[self.Pshotfid/0.1**3, self.Pshotfid/0.1**3, self.Pshotfid/0.1**3],
                                     [self.Pshotfid/0.1**2, self.Pshotfid/0.1**2, self.Pshotfid/0.1**2],
                                     [self.Pshotfid*0.1, self.Pshotfid*0.1, self.Pshotfid*0.1],
                                     [self.Pshotfid, self.Pshotfid, self.Pshotfid],
                                     [self.Pshotfid/0.1, self.Pshotfid/0.1, self.Pshotfid/0.1]], dtype='float')

        if hasattr(self, 'prior_inflation'):
            if self.prior_inflation:
                self.inflate_priors = 10.
            else:
                self.inflate_priors = 1.
        else:
            self.inflate_priors = 1.

        ## Define parameters for Gaussian quadrature
        n_gauss = 30  # number of Gaussian quadrature points
        [self.gauss_mu, self.gauss_w] = p_roots(n_gauss)

        self.k_vals = None
        self.Pk0_data = None
        self.Pk2_data = None
        self.Pk4_data = None

        for index_z in range(self.n_bin):
            data = np.loadtxt(os.path.join(self.data_directory, self.file[index_z]))
            # define input arrays
            if index_z == 0:
                self.k_vals = data[:, 0]
                self.Pk0_data = np.zeros((self.n_bin, len(data[:, 0])))
                self.Pk2_data = np.zeros((self.n_bin, len(data[:, 0])))
                self.Pk4_data = np.zeros((self.n_bin, len(data[:, 0])))

            self.Pk0_data[index_z] = data[:, 1]
            if self.use_quadrupole:
                self.Pk2_data[index_z] = data[:, 2]
            if self.use_hexadecapole:
                self.Pk4_data[index_z] = data[:, 3]

        self.muGrid, self.kGrid = np.meshgrid(self.gauss_mu, self.k_vals)
        self.wGrid = np.einsum('i,j->ij', np.ones(self.k_vals.shape), self.gauss_w)


        # Load in covariance matrices
        self.all_cov = np.zeros((self.n_bin, (1 + self.use_quadrupole + self.use_hexadecapole)*len(self.k_vals), (1 + self.use_quadrupole + self.use_hexadecapole)*len(self.k_vals)))

        zeros = np.zeros_like(self.k_vals, dtype='float')

        for index_z in range(self.n_bin):
            this_cov = np.loadtxt(os.path.join(self.data_directory, self.cov_file[index_z]))

            if self.use_quadrupole and self.use_hexadecapole:
                if len(this_cov) == len(self.k_vals) * 3:
                    pass
                else:
                    raise Exception('Need correct size covariance for monopole+quadrupole+hexadecapole analysis')
            elif self.use_quadrupole and self.use_hexadecapole:
                if len(this_cov) == len(self.k_vals) * 2:
                    pass
                elif len(this_cov) == len(self.k_vals) * 3:
                    this_cov = this_cov[:2*len(self.k_vals), :2*len(self.k_vals)]
                else:
                    raise Exception('Need correct size covariance for monopole+quadrupole analysis')
            else:
                if len(this_cov) == len(self.k_vals):
                    pass
                elif len(this_cov) == len(self.k_vals) * 2 or len(this_cov) == len(self.k_vals) * 3:
                    this_cov = this_cov[:len(self.k_vals), :len(self.k_vals)]
                else:
                    raise Exception('Need correct size covariance for monopole-only analysis')

            self.all_cov[index_z]=this_cov

        # Now define multipoles
        self.leg2 = legendre(2,self.muGrid)
        self.leg4 = legendre(4,self.muGrid)


        self.full_invcov = np.zeros(self.all_cov.shape)
        ## Compute theoretical error envelope and combine with usual covariance
        for index_z in range(self.n_bin):
            data = np.loadtxt(os.path.join(self.data_directory, self.file_fid[index_z]))

            P0 = data[:, 1]

            envelope_power0 = P0 * 2  # extra factor to ensure we don't underestimate error
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

            broadband_marg_matrix = np.zeros_like(cov_theoretical_error, dtype='float')

            if self.use_poly_marg:
                for n in range(-3, 2):
                    dtheory_da = self.k_vals**n
                    if self.use_quadrupole and self.use_hexadecapole:
                        dtheory0_da = np.hstack([dtheory_da, zeros, zeros])
                        dtheory2_da = np.hstack([zeros, dtheory_da, zeros])
                        dtheory4_da = np.hstack([zeros, zeros, dtheory_da])

                        broadband_marg_matrix += self.poly_priors[n + 3, 0, index_z]**2 + np.outer(dtheory0_da, dtheory0_da) \
                                                 + self.poly_priors[n + 3, 1, index_z]**2 + np.outer(dtheory2_da, dtheory2_da) \
                                                 + self.poly_priors[n + 3, 2, index_z]**2 + np.outer(dtheory4_da, dtheory4_da)
                    elif self.use_quadrupole:
                        dtheory0_da = np.hstack([dtheory_da, zeros])
                        dtheory2_da = np.hstack([zeros, dtheory_da])

                        broadband_marg_matrix += self.poly_priors[n + 3, 0, index_z]**2 + np.outer(dtheory0_da, dtheory0_da) \
                                                 + self.poly_priors[n + 3, 1, index_z]**2 + np.outer(dtheory2_da, dtheory2_da)

                    else:
                        broadband_marg_matrix += self.poly_priors[n + 3, 0, index_z]**2 + np.outer(dtheory_da, dtheory_da)

            self.full_invcov[index_z] = np.linalg.inv(cov_theoretical_error + self.all_cov[index_z] + broadband_marg_matrix)

    @staticmethod
    def evaluate_pt_model(all_theory, k, h, fz, norm, b1, b2, bG2, bGamma3, css0, css2, css4, a2, Pshot):
        theory0 = ((norm**2 * all_theory[15]
                    + norm**4 * (all_theory[21])
                    + norm**1 * b1 * all_theory[16]
                    + norm**3 * b1 * (all_theory[22])
                    + norm**0 * b1**2 * all_theory[17]
                    + norm**2 * b1**2 * all_theory[23]
                    + 0.25 * norm**2 * b2**2 * all_theory[1]
                    + b1 * b2 * norm**2 * all_theory[30]
                    + b2 * norm**3 * all_theory[31]
                    + b1 * bG2 * norm**2 * all_theory[32]
                    + bG2 * norm**3 * all_theory[33]
                    + b2 * bG2 * norm**2 * all_theory[4]
                    + bG2**2 * norm**2 * all_theory[5]
                    + 2. * css0 * norm**2 * all_theory[11] / h**2
                    + (2. * bG2 + 0.8 * bGamma3 * norm) * norm**2 * (b1 * all_theory[7] + norm * all_theory[8])) * h**3
                   + Pshot
                   # + a0*(k/0.45)**2.
                   + a2 * (1. / 3.) * (k / 0.45)**2.
                   # + fz**2*b4*k**2*(norm**2*fz**2/9. + 2.*fz*b1*norm/7. + b1**2/5)*(35./8.)*all_theory[13]*h
                   )

        # Quadrupole
        theory2 = ((norm**2 * all_theory[18]
                    + norm**4 * (all_theory[24])
                    + norm**1 * b1 * all_theory[19]
                    + norm**3 * b1 * (all_theory[25])
                    + b1**2 * norm**2 * all_theory[26]
                    + b1 * b2 * norm**2 * all_theory[34]
                    + b2 * norm**3 * all_theory[35]
                    + b1 * bG2 * norm**2 * all_theory[36]
                    + bG2 * norm**3 * all_theory[37]
                    + 2. * css2 * norm**2 * all_theory[12] / h**2
                    + (2. * bG2 + 0.8 * bGamma3 * norm) * norm**3 * all_theory[9]) * h**3
                   + a2 * (2. / 3.) * (k / 0.45)**2.
                   # + fz**2*b4*k**2*((norm**2*fz**2*70. + 165.*fz*b1*norm+99.*b1**2)*4./693.)*(35./8.)*all_theory[13]*h
                   )

        # Hexadecapole
        theory4 = ((norm**2 * all_theory[20]
                    + norm**4 * all_theory[27]
                    + b1 * norm**3 * all_theory[28]
                    + b1**2 * norm**2 * all_theory[29]
                    + b2 * norm**3 * all_theory[38]
                    + bG2 * norm**3 * all_theory[39]
                    + 2. * css4 * norm**2 * all_theory[13] / h**2) * h**3
                   # + fz**2*b4*k**2*(norm**2*fz**2*48./143. + 48.*fz*b1*norm/77.+8.*b1**2/35.)*(35./8.)*all_theory[13]*h
                   )

        return theory0, theory2, theory4

    def pk_model(self, z, h, alpha_perp, alpha_par, norm=1.0, alpha_rs=1.0, bias_params=None):
        F = alpha_par / alpha_perp

        k1 = self.kGrid / alpha_perp * np.sqrt(1. + np.power(self.muGrid, 2.) * (np.power(F, -2.) - 1.))
        mu1 = self.muGrid / (F * np.sqrt(1. + np.power(self.muGrid, 2.) * (np.power(F, -2.) - 1.)))

        all_theory = self.cosmo_fid.get_pk_mult(k1.flatten() * h, z, len(k1.flatten()), alpha_rs=alpha_rs)

        fz = self.cosmo_fid.scale_independent_growth_factor_f(z)
        P0,P2,P4 = self.evaluate_pt_model(all_theory, k1.flatten(), h, fz, norm, *bias_params)

        P0 = P0.reshape(k1.shape)
        P2 = P2.reshape(k1.shape)
        P4 = P4.reshape(k1.shape)

        if self.use_quadrupole and self.use_hexadecapole:
            P_k_mu = P0 * legendre(0, mu1) + P2 * legendre(2, mu1) + P4 * legendre(4, mu1)
        elif self.use_quadrupole:
            P_k_mu = P0 * legendre(0, mu1) + P2 * legendre(2, mu1)
        else:
            P_k_mu = P0 * legendre(0, mu1)


        P0_est = np.sum(P_k_mu * self.wGrid, axis=-1) / 2.

        if self.use_quadrupole:
            P2_est = np.sum(P_k_mu * self.leg2 * self.wGrid, axis=-1) * 5. / 2.
        else:
            P2_est = 0.

        if self.use_hexadecapole:
            P4_est = np.sum(P_k_mu * self.leg4 * self.wGrid, axis=-1) * 9. / 2.
        else:
            P4_est = 0.

        return P0_est, P2_est, P4_est


    def loglkl(self, cosmo, data):

        norm = data.mcmc_parameters['norm']['current'] * data.mcmc_parameters['norm']['scale']
        alpha_rs = data.mcmc_parameters['alpha_rs']['current'] * data.mcmc_parameters['alpha_rs']['scale']
        h=self.cosmo_fid.h()

        b1 = [data.mcmc_parameters['b1_%d' % i]['current'] * data.mcmc_parameters['b1_%d' % i]['scale'] for i in range(1, self.n_bin + 1)]
        b2 = [data.mcmc_parameters['b2_%d' % i]['current'] * data.mcmc_parameters['b2_%d' % i]['scale'] for i in range(1, self.n_bin + 1)]
        bG2 = [data.mcmc_parameters['bG2_%d' % i]['current'] * data.mcmc_parameters['bG2_%d' % i]['scale'] for i in range(1, self.n_bin + 1)]

        """a0m1 = [data.mcmc_parameters['a0m1_%d' % i]['current'] * data.mcmc_parameters['a0m1_%d' % i]['scale'] for i in range(1, self.n_bin + 1)]
        a00 = [data.mcmc_parameters['a00_%d' % i]['current'] * data.mcmc_parameters['a00_%d' % i]['scale'] for i in range(1, self.n_bin + 1)]
        a01 = [data.mcmc_parameters['a01_%d' % i]['current'] * data.mcmc_parameters['a01_%d' % i]['scale'] for i in range(1, self.n_bin + 1)]

        a1m1 = [data.mcmc_parameters['a1m1_%d' % i]['current'] * data.mcmc_parameters['a1m1_%d' % i]['scale'] for i in range(1, self.n_bin + 1)]
        a10 = [data.mcmc_parameters['a10_%d' % i]['current'] * data.mcmc_parameters['a10_%d' % i]['scale'] for i in range(1, self.n_bin + 1)]
        a11 = [data.mcmc_parameters['a11_%d' % i]['current'] * data.mcmc_parameters['a11_%d' % i]['scale'] for i in range(1, self.n_bin + 1)]

        a2m1 = [data.mcmc_parameters['a2m1_%d' % i]['current'] * data.mcmc_parameters['a2m1_%d' % i]['scale'] for i in range(1, self.n_bin + 1)]
        a20 = [data.mcmc_parameters['a20_%d' % i]['current'] * data.mcmc_parameters['a20_%d' % i]['scale'] for i in range(1, self.n_bin + 1)]
        a21 = [data.mcmc_parameters['a21_%d' % i]['current'] * data.mcmc_parameters['a21_%d' % i]['scale'] for i in range(1, self.n_bin + 1)]"""

        scale_0 = [data.mcmc_parameters['scale0_%d' % i]['current'] * data.mcmc_parameters['scale0_%d' % i]['scale'] for i in range(1, self.n_bin + 1)]
        scale_2 = [data.mcmc_parameters['scale2_%d' % i]['current'] * data.mcmc_parameters['scale2_%d' % i]['scale'] for i in range(1, self.n_bin + 1)]
        scale_4 = [data.mcmc_parameters['scale4_%d' % i]['current'] * data.mcmc_parameters['scale4_%d' % i]['scale'] for i in range(1, self.n_bin + 1)]

        b2sig = 1. * self.inflate_priors
        bG2sig = 1. * self.inflate_priors

        alpha_par = self.cosmo_fid.z_of_r(self.z)[1] * self.cosmo_fid.rs_drag() / (cosmo.z_of_r(self.z)[1] * cosmo.rs_drag())
        alpha_perp = cosmo.z_of_r(self.z)[0] * self.cosmo_fid.rs_drag() / (self.cosmo_fid.z_of_r(self.z)[0] * cosmo.rs_drag())

        chi2=0

        for z_i,z in enumerate(self.z):
            bias_params = (b1[z_i], b2[z_i], bG2[z_i], self.bGamma3fid[z_i], self.css0fid[z_i], self.css2fid[z_i], self.css4fid[z_i], 0.0, self.Pshotfid[z_i])

            P0,P2,P4 = self.pk_model(z, h, alpha_perp[z_i], alpha_par[z_i], norm, alpha_rs, bias_params)

            # Create vector of residual pk
            if self.use_quadrupole and self.use_hexadecapole:
                stacked_model = np.concatenate([scale_0[z_i] * P0, scale_2[z_i] * P2, scale_4[z_i] * P4])
                stacked_data = np.concatenate([self.Pk0_data[z_i], self.Pk2_data[z_i], self.Pk4_data[z_i]])
            elif self.use_quadrupole:
                stacked_model = np.concatenate([scale_0[z_i] * P0, scale_2[z_i] * P2])
                stacked_data = np.concatenate([self.Pk0_data[z_i], self.Pk2_data[z_i]])
            else:
                stacked_model = scale_0[z_i] * P0
                stacked_data = self.Pk0_data[z_i]

            resid_vec = stacked_data - stacked_model

            chi2 += float(np.matmul(resid_vec.T, np.matmul(self.full_invcov[z_i], resid_vec)))

            chi2 += (b2[z_i] - self.b2fid[z_i])**2. / b2sig**2. + (bG2[z_i] - self.bG2fid[z_i])**2. / bG2sig**2.

        if self.use_alpha_rs_prior:
            chi2 += (alpha_rs - 1.0)**2. / self.alpha_rs_prior**2.

        #print("chi2_euclidP=", chi2)
        loglkl = -0.5 * chi2

        return loglkl

