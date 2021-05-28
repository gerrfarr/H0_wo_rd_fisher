from montepython.likelihood_class import Likelihood_prior

class alpha_rs_priors(Likelihood_prior):

    # Add prior on sound horizon rescaling
    def __init__(self,path,data,command_line):

        Likelihood_prior.__init__(self,path,data,command_line)

    def loglkl(self, cosmo, data):

        alpha_rs = (data.mcmc_parameters['alpha_rs']['current'] *
                    data.mcmc_parameters['alpha_rs']['scale'])
        chi2 = (alpha_rs-self.mean)**2./self.std**2.

        loglkl = -0.5 * chi2

        return loglkl