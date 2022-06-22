import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pdb
from torch.utils.data import DataLoader
from training.train_conditional_adversarial_AE import TrainAdvAE
from models.conditional_adversarial_autoencoder import Generator, Critic, Encoder
from tqdm import tqdm
import pandas as pd
import hamiltorch
from utils.seed_everything import seed_everything
from data_handling.WNTR_dataloader import NetworkSensorDataset
from inference.posterior import LatentPosterior
from inference.compute_posterior import MAP_estimate, hamiltonian_monte_carlo
import ray
import time

torch.set_default_tensor_type(torch.DoubleTensor)

seed_everything(1)





if __name__ == '__main__':

    continue_training = False
    train = False
    cuda = False
    if cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

    sensor_locations = {'link': [4, 8, 17, 20, 27, 30, 33],
                        'node': [4, 9, 18, 20, 27, 28, 32]}
    noise = {'mean': 0, 'std': 0.05}

    num_skip_steps = 3
    memory = 108
    # Load data
    data_path = 'data/training_data_with_leak'
    sensor_dataset = NetworkSensorDataset(
        data_path=data_path,
        num_files=7500,
        memory=memory,
        num_skip_steps=num_skip_steps,
        sensor_locations=sensor_locations,
        noise=noise,
        transformer_state=None,
        with_pars=True
    )
    dataloader = DataLoader(
        sensor_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8
    )
    #lol, lal = sensor_dataset[0]
    #pdb.set_trace()

    num_out_channels = len(sensor_locations['link']) + len(sensor_locations['node'])
    latent_dim = 8

    # Fit MinMaxScaler to sensor_dataset in batches
    transformer = MinMaxScaler()
    #for i, (data, _) in enumerate(dataloader):
    for i in range(200):
        data, _ = next(iter(dataloader))
        np_data = data[:,:,:,0].view(-1, num_out_channels).numpy()
        transformer.partial_fit(np_data)
    sensor_dataset.transformer_state = transformer


    dataloader = DataLoader(
        sensor_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=8
    )

    generator = Generator(
        latent_dim=latent_dim,
        hidden_channels=[40, 32, 24, 16, num_out_channels],
        par_dim=34
    ).to(device)


    encoder = Encoder(
        latent_dim=latent_dim,
        hidden_channels=[num_out_channels, 16, 24, 32, 40],
    ).to(device)

    critic = Critic(
        latent_dim=latent_dim,
        hidden_neurons=[64, 64],
    ).to(device)

    encoder_recon_optimizer = torch.optim.RMSprop(encoder.parameters(), lr=1e-2, weight_decay=1e-8)
    generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=1e-2, weight_decay=1e-8)

    encoder_reg_optimizer = torch.optim.RMSprop(encoder.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.RMSprop(critic.parameters(), lr=1e-4)

    if continue_training:
        checkpoint = torch.load('AdvAE')
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])

        critic.load_state_dict(checkpoint['critic_state_dict'])
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder_recon_optimizer.load_state_dict(checkpoint['encoder_recon_optimizer_state_dict'])
        encoder_reg_optimizer.load_state_dict(checkpoint['encoder_reg_optimizer_state_dict'])

    if train:
        trainer = TrainAdvAE(
                generator=generator,
                critic=critic,
                encoder=encoder,
                encoder_recon_optimizer=encoder_recon_optimizer,
                critic_optimizer=critic_optimizer,
                generator_optimizer=generator_optimizer,
                encoder_reg_optimizer=encoder_reg_optimizer,
                n_critic=4,
                gamma=10,
                save_string='AdvAE',
                n_epochs=1000,
                device=device)
        trainer.train(dataloader=dataloader)
        generator.eval()
    else:
        checkpoint = torch.load('AdvAE')
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder.eval()


    '''
    z = []
    for i, (data, _) in enumerate(dataloader):
        data = data.view(-1, data.size(-2), data.size(-1))
        data = data.to(device)
        z_enc = encoder(data)
        z.append(z_enc)
        if i == 20:
            break
    z = torch.cat(z, dim=0)
    z = z.detach().cpu().numpy()
    plt.figure()
    plt.scatter(z[:, 0], z[:, 1], s=1)
    plt.show()
    pdb.set_trace()
    '''



    # Test
    timing = []
    ray.init(num_cpus=17)
    plt.figure(figsize=(10, 10))
    for case in range(1):
        test_data = pd.read_pickle('data/test_data_with_leak/network_'+str(case))
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

        HMC_params = {'num_samples': 500,
                      'step_size': 1.,
                      'num_steps_per_sample': 5,
                      'burn': 200,
                      'integrator': hamiltorch.Integrator.IMPLICIT,
                      'sampler': hamiltorch.Sampler.HMC_NUTS,
                      'desired_accept_rate': 0.3
                      }

        @ray.remote
        def reconstruct_observations(observations, pars):
            reconstructed_data_i = []
            reconstructed_std_i = []
            for i in range(0, observations.shape[1], memory):

                obs = observations[:, i:i+memory]
                if obs.shape[1] < memory:
                    break

                z = encoder(obs.unsqueeze(0))
                gen_state = generator(z, pars)[0].detach().cpu().numpy()
                reconstructed_data_i.append(gen_state)

                latent_posterior.generator = lambda x: generator(x, pars)
                z_samples = hamiltonian_monte_carlo(
                        z_init=z[0],
                        posterior=latent_posterior,
                        observations=obs,
                        HMC_params=HMC_params,
                        print_progress=False
                )
                gen_samples = generator(z_samples, pars.repeat(z_samples.shape[0], 1))
                gen_std = gen_samples.std(dim=0).detach().cpu().numpy()
                reconstructed_std_i.append(gen_std)
                '''
                reconstructed_std_i.append(np.array([[1.]]))
                '''

            reconstructed_data_i = np.concatenate(reconstructed_data_i, axis=1)
            reconstructed_std_i = np.concatenate(reconstructed_std_i, axis=1)

            return {'estimate': reconstructed_data_i,
                    'std': reconstructed_std_i}


        t0 = time.time()
        reconstruction = []
        z_MAP = torch.randn((1, latent_dim), requires_grad=True, device=device)
        for j in range(34):
            pars = torch.zeros(1, 34, device=device)
            pars[0, j] = 1

            reconstruction.append(reconstruct_observations.remote(observations, pars))
            #reconstruction.append(reconstruct_observations(observations, pars))

        reconstruction = ray.get(reconstruction)

        t1 = time.time()
        timing.append(t1-t0)

        #gen_MAP = generator(z_MAP)[0].detach().cpu().numpy()
        observations = observations.detach().cpu().numpy()
        observations = observations[:, 0:reconstruction[0]['estimate'].shape[1]]

        error = []
        std = []
        for i in range(len(reconstruction)):
            err = np.linalg.norm(reconstruction[i]['estimate'] - observations)
            error.append(err)

            std.append(np.linalg.norm(reconstruction[i]['std']))

        plt.subplot(3, 3, case+1)
        plt.plot(range(1, 35), error, '.-', label='error', markersize=10)
        plt.plot(range(1, 35), std, '.-', label='std', markersize=10)
        plt.axvline(x=leak['pipe'], color='black', label='leak pipe')
        plt.grid()
        plt.legend()
        plt.xlabel('Pipe section')
        plt.ylabel('Error')
    plt.show()
    print(np.mean(timing))




    '''
    t_vec = np.linspace(0, 2*60*60*24, observations.shape[1])

    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plt.figure(figsize=(20, 15))
    plt.subplot(2, 2, 1)
    for i in range(0, len(reconstructed_data)):
        t_vec_idx = np.arange(i * memory, (i + 1) * memory)

        gen_state = reconstructed_data[i]
        true_state = observations[:, i * memory:i * memory + memory]
        error = np.abs(true_state - gen_state) / np.sum(np.abs(true_state))

        # std = np.linalg.norm(reconstructed_std[i], axis=0)

        plt.axvline(x=t_vec[t_vec_idx[-1]], color='k', linestyle='--',
                    linewidth=0.5, alpha=0.3)
        for j in range(error.shape[0]//2):
            plt.plot(t_vec[t_vec_idx], error[j],
                     color=color_list[j],
                     linewidth=0.5,)
        # plt.plot(t_vec[t_vec_idx], std, color='tab:green', linewidth=0.5)
    plt.axvline(x=leak['start_time'], color='k', linestyle='-', linewidth=2)
    plt.title('Flowrate Reconstruction Error')
    plt.legend(sensor_locations['link'])

    plt.subplot(2, 2, 2)
    for i in range(0, len(reconstructed_data)):
        t_vec_idx = np.arange(i * memory, (i + 1) * memory)

        gen_state = reconstructed_data[i]
        true_state = observations[:, i * memory:i * memory + memory]
        error = np.abs(true_state - gen_state) / np.sum(np.abs(true_state))

        # std = np.linalg.norm(reconstructed_std[i], axis=0)

        plt.axvline(x=t_vec[t_vec_idx[-1]], color='k', linestyle='--',
                    linewidth=0.5, alpha=0.3)
        for j in range(error.shape[0]//2):
            plt.plot(t_vec[t_vec_idx], error[j+7],
                     color=color_list[j],
                     linewidth=0.5,)
        # plt.plot(t_vec[t_vec_idx], std, color='tab:green', linewidth=0.5)
    plt.axvline(x=leak['start_time'], color='k', linestyle='-', linewidth=2)
    plt.legend(sensor_locations['node'])
    plt.title('Head Reconstruction Error')

    plt.subplot(2, 2, 3)
    for i in range(0, len(reconstructed_data)):
        t_vec_idx = np.arange(i * memory, (i + 1) * memory)

        std = reconstructed_std[i]

        # std = np.linalg.norm(reconstructed_std[i], axis=0)

        plt.axvline(x=t_vec[t_vec_idx[-1]], color='k', linestyle='--',
                    linewidth=0.5, alpha=0.3)
        for j in range(std.shape[0]//2):
            plt.plot(t_vec[t_vec_idx], std[j],
                     color=color_list[j],
                     linewidth=0.5,)
        # plt.plot(t_vec[t_vec_idx], std, color='tab:green', linewidth=0.5)
    plt.axvline(x=leak['start_time'], color='k', linestyle='-', linewidth=2)
    plt.legend(sensor_locations['link'])
    plt.title('Standard deviation of flowrate')

    plt.subplot(2, 2, 4)
    for i in range(0, len(reconstructed_data)):
        t_vec_idx = np.arange(i * memory, (i + 1) * memory)

        std = reconstructed_std[i]

        # std = np.linalg.norm(reconstructed_std[i], axis=0)

        plt.axvline(x=t_vec[t_vec_idx[-1]], color='k', linestyle='--',
                    linewidth=0.5, alpha=0.3)
        for j in range(std.shape[0]//2):
            plt.plot(t_vec[t_vec_idx], std[j+7],
                     color=color_list[j],
                     linewidth=0.5,)
        # plt.plot(t_vec[t_vec_idx], std, color='tab:green', linewidth=0.5)
    plt.axvline(x=leak['start_time'], color='k', linestyle='-', linewidth=2)
    plt.legend(sensor_locations['node'])
    plt.title('Standard deviation of head')
    plt.show()


    pdb.set_trace()
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    for i in range(0, len(reconstructed_data)):
        t_vec_idx = np.arange(i*memory, (i+1)*memory)

        gen_state = reconstructed_data[i]
        true_state = observations[:, i*memory:i*memory+memory]
        error = np.sum(np.abs(true_state - gen_state), axis=0)/np.sum(np.abs(true_state), axis=0)

        #std = np.linalg.norm(reconstructed_std[i], axis=0)

        plt.axvline(x=t_vec[t_vec_idx[-1]], color='k', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.plot(t_vec[t_vec_idx], true_state[2], color='tab:blue')
        plt.plot(t_vec[t_vec_idx], gen_state[2], color='tab:orange', linewidth=0.5)
        plt.plot(t_vec[t_vec_idx], error, color='tab:green', linewidth=0.5)
        #plt.plot(t_vec[t_vec_idx], std, color='tab:green', linewidth=0.5)
    plt.axvline(x=leak['start_time'], color='k', linestyle='-', linewidth=2)

    plt.subplot(2, 2, 2)
    for i in range(0, len(reconstructed_data)):
        t_vec_idx = np.arange(i*memory, (i+1)*memory)

        gen_state = reconstructed_data[i]
        true_state = observations[:, i*memory:i*memory+memory]
        error = np.sum(np.abs(true_state - gen_state), axis=0)/np.sum(np.abs(true_state), axis=0)

        #std = np.linalg.norm(reconstructed_std[i], axis=0)

        plt.axvline(x=t_vec[t_vec_idx[-1]], color='k', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.plot(t_vec[t_vec_idx], true_state[5], color='tab:blue')
        plt.plot(t_vec[t_vec_idx], gen_state[5], color='tab:orange', linewidth=0.5)
        plt.plot(t_vec[t_vec_idx], error, color='tab:green', linewidth=0.5)
        #plt.plot(t_vec[t_vec_idx], std, color='tab:green', linewidth=0.5)
    plt.axvline(x=leak['start_time'], color='k', linestyle='-', linewidth=2)

    plt.subplot(2, 2, 3)
    for i in range(0, len(reconstructed_data)):
        t_vec_idx = np.arange(i*memory, (i+1)*memory)

        gen_state = reconstructed_data[i]
        true_state = observations[:, i*memory:i*memory+memory]
        error = np.sum(np.abs(true_state - gen_state), axis=0)/np.sum(np.abs(true_state), axis=0)

        #std = np.linalg.norm(reconstructed_std[i], axis=0)

        plt.axvline(x=t_vec[t_vec_idx[-1]], color='k', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.plot(t_vec[t_vec_idx], true_state[10], color='tab:blue')
        plt.plot(t_vec[t_vec_idx], gen_state[10], color='tab:orange', linewidth=0.5)
        plt.plot(t_vec[t_vec_idx], error, color='tab:green', linewidth=0.5)
        #plt.plot(t_vec[t_vec_idx], std, color='tab:green', linewidth=0.5)
    plt.axvline(x=leak['start_time'], color='k', linestyle='-', linewidth=2)

    plt.subplot(2, 2, 4)
    for i in range(0, len(reconstructed_data)):
        t_vec_idx = np.arange(i*memory, (i+1)*memory)

        gen_state = reconstructed_data[i]
        true_state = observations[:, i*memory:i*memory+memory]
        error = np.sum(np.abs(true_state - gen_state), axis=0)/np.sum(np.abs(true_state), axis=0)

        #std = np.linalg.norm(reconstructed_std[i], axis=0)

        plt.axvline(x=t_vec[t_vec_idx[-1]], color='k', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.plot(t_vec[t_vec_idx], true_state[13], color='tab:blue')
        plt.plot(t_vec[t_vec_idx], gen_state[13], color='tab:orange', linewidth=0.5)
        plt.plot(t_vec[t_vec_idx], error, color='tab:green', linewidth=0.5)
        #plt.plot(t_vec[t_vec_idx], std, color='tab:green', linewidth=0.5)
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
    '''

