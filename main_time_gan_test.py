import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pdb
from torch.utils.data import DataLoader
from training.train_time_gan import TrainTimeGAN
from models.time_gan import Generator, Critic
from tqdm import tqdm
import pandas as pd
import hamiltorch
from utils.seed_everything import seed_everything
from data_handling.WNTR_dataloader import NetworkSensorDataset
from inference.posterior import LatentPosterior
from inference.compute_posterior import MAP_estimate, hamiltonian_monte_carlo

torch.set_default_tensor_type(torch.DoubleTensor)

seed_everything(1)


if __name__ == '__main__':

    continue_training = True
    train = True
    cuda = True
    if cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

    sensor_locations = {'link': [4, 8, 17, 20, 27, 30, 33],
                        'node': [4, 9, 18, 20, 27, 28, 32]}
    noise = {'mean': 0, 'std': 0.005}

    num_skip_steps = 3
    memory = 108
    # Load data
    data_path = 'data/training_data_no_leak'
    sensor_dataset = NetworkSensorDataset(
        data_path=data_path,
        num_files=7000,
        memory=memory,
        num_skip_steps=num_skip_steps,
        sensor_locations=sensor_locations,
        noise=noise,
        transformer_state=None,
    )
    dataloader = DataLoader(
        sensor_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=8
    )

    num_out_channels = len(sensor_locations['link']) + len(sensor_locations['node'])
    latent_dim = 32

    # Fit MinMaxScaler to sensor_dataset in batches
    transformer = MinMaxScaler()
    for i, data in enumerate(dataloader):
        np_data = data[:,:,:,0].view(-1, num_out_channels).numpy()
        transformer.partial_fit(np_data)
    sensor_dataset.transformer_state = transformer

    dataloader = DataLoader(
        sensor_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=12
    )

    generator = Generator(
        latent_dim=latent_dim,
        hidden_channels=[64, 48, 32, 16, num_out_channels],
    ).to(device)


    critic = Critic(
        hidden_channels=[num_out_channels, 16, 32, 48, 64],
    ).to(device)

    generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.RMSprop(critic.parameters(), lr=1e-4)

    if continue_training:
        checkpoint = torch.load('TimeGAN')
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])

        critic.load_state_dict(checkpoint['critic_state_dict'])
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

    if train:
        trainer = TrainTimeGAN(
                generator=generator,
                critic=critic,
                generator_optimizer=generator_optimizer,
                critic_optimizer=critic_optimizer,
                n_critic=3,
                gamma=10,
                save_string='TimeGAN',
                n_epochs=5000,
                device=device)
        trainer.train(dataloader=dataloader)
        generator.eval()
    else:
        checkpoint = torch.load('TimeGAN')
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()

    # Test
    test_data = pd.read_pickle('data/test_data_with_leak/network_8')
    leak = test_data['leak']
    flow_rate = test_data['flow_rate'][sensor_locations['link']]
    head = test_data['head'][sensor_locations['node']]

    test_data = np.concatenate((flow_rate, head), axis=1)
    test_data = test_data[0::num_skip_steps]

    std = noise['std'] * np.abs(test_data)
    test_data += std * np.random.randn(test_data.shape[0], test_data.shape[1])

    test_data = transformer.transform(test_data)
    std = std.mean(axis=0)
    noise_std = np.repeat(std.reshape(-1, 1), memory, axis=1)
    noise_mean = np.zeros(noise_std.shape)
    data_noise = {
        'mean': torch.tensor(noise_mean, device=device),
        'std': torch.tensor(noise_std, device=device)
    }
    observations = torch.tensor(test_data, device=device)
    observations = np.swapaxes(observations, 0, 1)
    #observations = observations[:, 100:100+memory]

    latent_posterior = LatentPosterior(
        generator=generator,
        data_noise=data_noise,
        device=device,
    )

    HMC_params = {'num_samples': 2000,
                  'step_size': 1.,
                  'num_steps_per_sample': 5,
                  'burn': 1250,
                  'integrator': hamiltorch.Integrator.IMPLICIT,
                  'sampler': hamiltorch.Sampler.HMC_NUTS,
                  'desired_accept_rate': 0.3
                  }

    reconstructed_data = []
    reconstructed_std = []
    z_MAP = torch.randn((1, latent_dim), requires_grad=True, device=device)
    for i in range(0, observations.shape[1], memory):

        obs = observations[:, i:i+memory]
        if obs.shape[1] < memory:
            break

        z_MAP = MAP_estimate(
                z=z_MAP,
                observations=obs,
                posterior=latent_posterior,
                print_progress=True,
                num_iter=2500
        )

        z_samples = hamiltonian_monte_carlo(
                z_init=z_MAP[0],
                posterior=latent_posterior,
                observations=obs,
                HMC_params=HMC_params,
                print_progress=True
        )

        gen_samples = generator(z_samples)
        gen_mean = gen_samples.mean(dim=0).detach().cpu().numpy()
        gen_std = gen_samples.std(dim=0).detach().cpu().numpy()

        gen_state = generator(z_MAP)[0].detach().cpu().numpy()
        #reconstructed_data.append(gen_state)
        reconstructed_data.append(gen_mean)
        reconstructed_std.append(gen_std)

    #gen_MAP = generator(z_MAP)[0].detach().cpu().numpy()
    observations = observations.detach().cpu().numpy()

    t_vec = np.linspace(0, 2*60*60*24, observations.shape[1])

    plt.figure()
    for i in range(0, len(reconstructed_data)):
        t_vec_idx = np.arange(i*memory, (i+1)*memory)

        gen_state = reconstructed_data[i]
        true_state = observations[:, i*memory:i*memory+memory]
        error = np.sum(np.abs(true_state - gen_state), axis=0)/np.sum(np.abs(true_state), axis=0)

        std = np.linalg.norm(reconstructed_std[i], axis=0)

        plt.axvline(x=t_vec[t_vec_idx[-1]], color='k', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.plot(t_vec[t_vec_idx], true_state[6], color='tab:blue')
        plt.plot(t_vec[t_vec_idx], gen_state[6], color='tab:orange', linewidth=0.5)
        plt.plot(t_vec[t_vec_idx], error, color='tab:green', linewidth=0.5)
        plt.plot(t_vec[t_vec_idx], std, color='tab:green', linewidth=0.5)

    plt.axvline(x=leak['start_time'], color='k', linestyle='-', linewidth=2)
    plt.legend()
    plt.show()
    pdb.set_trace()

    plt.figure()
    plt.plot(observations[2, 0:memory], label='Observed', color='tab:blue')
    plt.plot(observations[5, 0:memory], color='tab:blue')
    plt.plot(observations[12, 0:memory], color='tab:blue')
    plt.plot(gen_MAP[2, :], label='GAN', color='tab:orange')
    plt.plot(gen_MAP[5, :], color='tab:orange')
    plt.plot(gen_MAP[12, :], color='tab:orange')
    plt.legend()
    plt.show()
    pdb.set_trace()



    posterior_params = {'generator': generator,
                        'obs_operator': lambda x: obs_operator(x, obs_idx),
                        'observations': observations,
                        'prior_mean': torch.zeros(latent_dim, device=device),
                        'prior_std': torch.ones(latent_dim, device=device),
                        'noise_mean': noise_mean,
                        'noise_std': noise_std
                        }
    HMC_params = {'num_samples': 5000,
                  'step_size': 1.,
                  'num_steps_per_sample': 5,
                  'burn': 3500,
                  'integrator': hamiltorch.Integrator.IMPLICIT,
                  'sampler': hamiltorch.Sampler.HMC_NUTS,
                  'desired_accept_rate': 0.3
                  }


    z_samples = hamiltonian_MC(
            z_init=z_MAP[0],
            posterior_params=posterior_params,
            HMC_params=HMC_params
        )

    gen_samples, gen_pars = generator(z_samples, output_pars=True)
    gen_mean = gen_samples.mean(dim=0).detach().cpu().numpy()
    gen_std = gen_samples.std(dim=0).detach().cpu().numpy()

    gen_pars = gen_pars.detach().cpu().numpy()

    gen_data = generator(z_MAP)
    plt.figure(figsize=(20,12))
    for i in range(3):
        plt.subplot(2, 3, i+1)
        plt.plot(gen_mean[i,:], label='Mean', color='red', linewidth=2)
        plt.fill_between(range(memory),
                         gen_mean[i,:] - 2*gen_std[i,:],
                         gen_mean[i,:] + 2*gen_std[i,:],
                         alpha=0.2,
                         color='red')
        plt.plot(list(obs_idx), observations[i].detach().cpu().numpy(),
                 '.b', markersize=10)
        plt.plot(data[0,i,:].numpy(), label='True', color='black', linewidth=2)
        plt.legend()
    for i in range(3):
        plt.subplot(2, 3, i+4)
        plt.hist(gen_pars[:,i], bins=50)
        plt.axvline(true_par[0,i], color='black', linewidth=2)

    plt.show()

