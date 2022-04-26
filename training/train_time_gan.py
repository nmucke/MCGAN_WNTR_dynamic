import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm


class TrainTimeGAN():
    def __init__(self, generator, critic,
                 generator_optimizer, critic_optimizer,
                 n_critic=5, gamma=10, save_string='TimeGAN',
                 n_epochs=100, device='cpu'):

        self.device = device
        self.generator = generator
        self.critic = critic
        self.generator_optimizer = generator_optimizer
        self.critic_optimizer = critic_optimizer

        self.eps = 1e-15

        self.n_epochs = n_epochs
        self.save_string = save_string

        self.generator.train(mode=True)
        self.critic.train(mode=True)

        self.latent_dim = self.generator.latent_dim
        self.n_critic = n_critic
        self.gamma = gamma

    def train(self, dataloader):
        """Train generator and critic"""

        generator_loss_list = []
        critic_loss_list = []

        pbar = tqdm(range(self.n_epochs),
                        total=self.n_epochs,
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for epoch in pbar:#range(1, self.n_epochs + 1):

            # Train one step
            generator_loss, critic_loss, gp = self.train_epoch(dataloader)

            #print(f'Epoch: {epoch}, generator_loss: {generator_loss:.5f}, ', end=' ')
            #print(f'critic_loss: {critic_loss:.5f}, ', end=' ')
            #print(f'gp_loss: {gp:.5f}, ', end=' ')

            pbar.set_postfix({"generator_loss": generator_loss,
                              "critic_loss": critic_loss,
                              "gp_loss": gp})

            # Save loss
            generator_loss_list.append(generator_loss)
            critic_loss_list.append(critic_loss)

            # Save generator and critic weights
            torch.save({
                'generator_state_dict': self.generator.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                }, self.save_string)

        # Save generator and critic weights

        torch.save({
                'generator_state_dict': self.generator.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            }, self.save_string)

        self.generator.train(mode=False)
        self.critic.train(mode=False)

        return generator_loss_list, critic_loss_list

    def train_epoch(self, dataloader):
        """Train generator and critic for one epoch"""

        #for bidx, real_data in tqdm(enumerate(dataloader),
        #         total=int(len(dataloader.dataset)/dataloader.batch_size)):
        for bidx, real_data in enumerate(dataloader):

            real_data = real_data.view(-1, real_data.size(-2), real_data.size(-1))

            shuffle_idx = torch.randperm(real_data.size(0))
            real_data = real_data[shuffle_idx]
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)

            critic_loss, gp = self.critic_train_step(real_data,
                                                     batch_size)

            # Train generator
            if bidx % self.n_critic == 0:
                generator_loss = self.generator_train_step(batch_size)

        return generator_loss, critic_loss, gp



    def critic_train_step(self, real_data, batch_size):
        """Train critic one step"""

        self.generator.eval()
        self.critic_optimizer.zero_grad()

        generated_data = self.sample(batch_size)

        grad_penalty = self.gradient_penalty(real_data, generated_data)
        cri_loss = self.critic(generated_data).mean() \
                 - self.critic(real_data).mean() + grad_penalty

        cri_loss.backward()
        self.critic_optimizer.step()

        self.generator.train(mode=True)

        return cri_loss.detach().item(),  grad_penalty.detach().item()



    def generator_train_step(self, batch_size):
        self.critic.eval()
        self.generator_optimizer.zero_grad()

        generated_data = self.sample(batch_size)

        generator_loss = -self.critic(generated_data).mean()
        generator_loss.backward()
        self.generator_optimizer.step()

        self.critic.train(mode=True)

        return generator_loss.detach().item()

    def gradient_penalty(self, data=None, generated_data=None):
        """Compute gradient penalty"""

        batch_size = data.size(0)
        epsilon = torch.rand(batch_size, data.size(1), 1, device=self.device)
        epsilon = epsilon.expand_as(data)

        interpolation = epsilon * data.data + (1 - epsilon) * generated_data
        interpolation = torch.autograd.Variable(interpolation,
                                                requires_grad=True)
        interpolation_critic_score = self.critic(interpolation)

        grad_outputs = torch.ones(interpolation_critic_score.size(),
                                  device=self.device)

        gradients = torch.autograd.grad(outputs=interpolation_critic_score,
                                        inputs=interpolation,
                                        grad_outputs=grad_outputs,
                                        create_graph=True,
                                        retain_graph=True)[0]
        gradients_norm = torch.sqrt(
            torch.sum(gradients ** 2, dim=1) + 1e-12)
        return self.gamma * ((gradients_norm - 1) ** 2).mean()

    def sample(self, n_samples):
        """Generate n_samples fake samples"""
        z = torch.randn(n_samples, self.latent_dim).to(self.device)
        return self.generator(z)
