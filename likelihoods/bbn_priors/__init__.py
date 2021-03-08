from montepython.likelihood_class import Likelihood_prior

class bbn_priors(Likelihood_prior):

    # Add priors to omega_b from BBN.
    def __init__(self,path,data,command_line):

        Likelihood_prior.__init__(self,path,data,command_line)

    def loglkl(self, cosmo, data):

        omb = (data.mcmc_parameters['omega_b']['current'] *
                    data.mcmc_parameters['omega_b']['scale'])
        chi2 = (omb-self.omb_mean)**2./self.omb_std**2.

        loglkl = -0.5 * chi2

        return loglkl