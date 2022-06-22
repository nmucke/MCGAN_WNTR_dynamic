import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import pdb

class LatentPosterior():

    def __init__(self, generator, data_noise, device, pars=None):
        self.generator = generator
        self.data_noise = data_noise
        self.device = device
        self.pars = pars

        self.noise_mean = data_noise['mean']
        self.noise_std = data_noise['std']

        self.latent_dim = self.generator.latent_dim
        self.latent_mean = torch.zeros(self.latent_dim, device=device)
        self.latent_std = torch.ones(self.latent_dim, device=device)
        self.latent_prior_dist = torch.distributions.Normal(
                self.latent_mean,
                self.latent_std
        )
    def log_prior(self, z):
        return self.latent_prior_dist.log_prob(z).sum()

    def log_likelihood(self, z, observations, pars=None):
        """
        Compute the likelihood of the data given the latent distribution z.
        """

        #if pars is not None:
        #    gen_state = self.generator(z.view(1, self.latent_dim), self.pars)
        #else:
        gen_state = self.generator(z.view(1, self.latent_dim))
        error = observations - gen_state[0]

        log_like = torch.distributions.Normal(
                self.noise_mean,
                self.noise_std
        ).log_prob(error).sum()
        return log_like

    def log_posterior(self, z, observations):
        """
        Compute the log posterior of the latent distribution z.
        """

        log_prior = self.log_prior(z)
        log_like = self.log_likelihood(z, observations)

        return log_prior + log_like
