import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import pdb
import os, sys
import hamiltorch

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def MAP_estimate(z, observations, posterior,
                 num_iter=2000, print_progress=False):
    optimizer = torch.optim.Adam([z], lr=.1, weight_decay=1e-5)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.85)

    if print_progress:
        pbar = tqdm(
            range(num_iter),
            total=num_iter,
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    else:
        pbar = range(num_iter)
    for _ in pbar:
        optimizer.zero_grad()
        log_posterior = -posterior.log_posterior(z, observations)
        #log_posterior = torch.norm(posterior.generator(z) - observations)
        log_posterior.backward()
        optimizer.step()

        #scheduler.step()

        if print_progress:
            pbar.set_postfix({"Error": log_posterior.item()})

    return z

def hamiltonian_monte_carlo(z_init,
                            observations,
                            posterior,
                            HMC_params,
                            print_progress=False):

        log_posterior = lambda z: posterior.log_posterior(z, observations)

        if print_progress:
            z_samples = hamiltorch.sample(log_prob_func=log_posterior,
                                          params_init=z_init,
                                          **HMC_params)
        else:
            with HiddenPrints():
                z_samples = hamiltorch.sample(log_prob_func=log_posterior,
                                              params_init=z_init,
                                              **HMC_params)
        return torch.stack(z_samples)
