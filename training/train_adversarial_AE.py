import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm


class TrainAdvAE():
    def __init__(self, generator, critic, encoder,
                 encoder_recon_optimizer, critic_optimizer, generator_optimizer,
                 encoder_reg_optimizer, n_critic=5, gamma=10,
                 save_string='AdvAE', n_epochs=100, device='cpu'):

        self.device = device
        self.generator = generator
        self.critic = critic
        self.encoder = encoder
        self.encoder_recon_optimizer = encoder_recon_optimizer
        self.encoder_reg_optimizer = encoder_reg_optimizer
        self.critic_optimizer = critic_optimizer
        self.generator_optimizer = generator_optimizer

        self.eps = 1e-15

        self.n_epochs = n_epochs
        self.save_string = save_string

        self.generator.train(mode=True)
        self.critic.train(mode=True)
        self.encoder.train(mode=True)

        self.latent_dim = self.generator.latent_dim
        self.n_critic = n_critic
        self.gamma = gamma

        self.recon_loss_function = nn.MSELoss()

    def train(self, dataloader):
        """Train generator and critic"""

        #pbar = tqdm(range(self.n_epochs),
        #                total=self.n_epochs,
        #                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for epoch in range(1, self.n_epochs + 1):

            # Train one step
            recon_loss, reg_loss, critic_loss, gp = self.train_epoch(dataloader)

            print(f'Epoch: {epoch}, recon_loss: {recon_loss:.5f}, ', end=' ')
            print(f'reg_loss: {reg_loss:.5f}, ', end=' ')
            print(f'critic_loss: {critic_loss:.5f}, ', end=' ')
            print(f'gp: {gp:.5f}')

            #pbar.set_postfix({"recon_loss": recon_loss,
            #                  "reg_loss": reg_loss,
            #                  "critic_loss": critic_loss,
            #                  "gp_loss": gp})

            # Save generator and critic weights
            torch.save({
                'generator_state_dict': self.generator.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'encoder_state_dict': self.encoder.state_dict(),
                'encoder_recon_optimizer_state_dict': self.encoder_recon_optimizer.state_dict(),
                'encoder_reg_optimizer_state_dict': self.encoder_reg_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
                }, self.save_string)

        # Save generator and critic weights
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'encoder_recon_optimizer_state_dict': self.encoder_recon_optimizer.state_dict(),
            'encoder_reg_optimizer_state_dict': self.encoder_reg_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            }, self.save_string)

        self.generator.eval()
        self.critic.eval()
        self.encoder.eval()

    def train_epoch(self, dataloader):
        """Train generator and critic for one epoch"""

        pbar = tqdm(
                enumerate(dataloader),
                total=int(len(dataloader.dataset)/dataloader.batch_size)
        )
        #for bidx, real_data in
        for bidx, real_data in pbar:

            real_data = real_data.view(-1, real_data.size(-2), real_data.size(-1))

            shuffle_idx = torch.randperm(real_data.size(0))
            real_data = real_data[shuffle_idx]
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)

            # Train critic
            for i in range(self.n_critic):
                critic_loss, gp = self.critic_train_step(real_data,batch_size)

            # reconstruction loss
            recon_loss = self.recon_train_step(real_data)

            # encoder regularization loss
            reg_loss = self.reg_train_step(real_data)


        return recon_loss, reg_loss, critic_loss, gp

    def recon_train_step(self, real_data):

        self.encoder_recon_optimizer.zero_grad()
        self.generator_optimizer.zero_grad()

        z_fake = self.encoder(real_data)
        fake_data = self.generator(z_fake)

        recon_loss = self.recon_loss_function(fake_data, real_data)
        recon_loss.backward()

        self.encoder_recon_optimizer.step()
        self.generator_optimizer.step()

        return recon_loss.detach().item()


    def reg_train_step(self, real_data):

        self.critic.eval()

        self.encoder_reg_optimizer.zero_grad()

        z_fake = self.encoder(real_data)
        reg_loss = -self.critic(z_fake).mean()
        reg_loss.backward()
        self.encoder_reg_optimizer.step()

        self.critic.train()

        return reg_loss.detach().item()

    def critic_train_step(self, real_data, batch_size):
        """Train critic one step"""

        self.encoder.eval()
        self.critic_optimizer.zero_grad()

        z_fake = self.encoder(real_data)
        z_real = self.sample(batch_size)

        grad_penalty = self.gradient_penalty(z_real, z_fake)
        critic_loss = self.critic(z_fake).mean() \
                    - self.critic(z_real).mean() + grad_penalty

        critic_loss.backward()
        self.critic_optimizer.step()

        self.encoder.train()

        return critic_loss.detach().item(),  grad_penalty.detach().item()

    def gradient_penalty(self, data=None, generated_data=None):
        """Compute gradient penalty"""

        batch_size = data.size(0)
        epsilon = torch.rand(batch_size, data.size(1), device=self.device)
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
        return z
