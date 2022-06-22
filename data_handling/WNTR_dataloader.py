import pdb
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

class NetworkSensorDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_path,
             num_files=10,
             num_skip_steps=10,
             memory=10,
             noise=None,
             sensor_locations=None,
             transformer_state=None,
             with_pars=False
             ):

        self.data_path_state = data_path
        self.num_files = num_files
        self.transformer_state = transformer_state
        self.num_skip_steps = num_skip_steps
        self.memory = memory
        self.sensor_locations = sensor_locations
        self.noise = noise
        self.transformer_state = transformer_state
        self.with_pars = with_pars

        self.state_IDs = [i for i in range(self.num_files)]

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        data_dict = pd.read_pickle(self.data_path_state + '/network_' + str(idx))
        flow_rate = data_dict['flow_rate'][self.sensor_locations['link']]
        head = data_dict['head'][self.sensor_locations['node']]
        #demand = data['demand'][self.sensor_locations['node']]

        #state = np.concatenate((flow_rate, head, demand), axis=1)
        state = np.concatenate((flow_rate, head), axis=1)
        state = state[0::self.num_skip_steps]

        #std = self.noise['std']*np.abs(state)
        #state += std*np.random.randn(state.shape[0], state.shape[1])

        if self.transformer_state is not None:
            state = self.transformer_state.transform(state)

        data = np.zeros((state.shape[0]-self.memory, self.memory, state.shape[1]))
        for i in range(state.shape[0]-self.memory):
            data[i] = state[i:i+self.memory, :]
        data = np.swapaxes(data, 1, 2)

        if self.with_pars:
            leak_pipe = data_dict['leak']['pipe']
            pars = torch.zeros(34)
            pars[leak_pipe-1] = 1
            pars = pars.unsqueeze(0)
            pars = pars.repeat(data.shape[0], 1)

            return data, pars
        else:
            return data






def get_dataloader(data_path,
                    num_files=100000,
                    transformer_state=None,
                    transformer_pars=None,
                    batch_size=512,
                    shuffle=True,
                    num_workers=2,
                    drop_last=True,
                    num_states_pr_sample=10,
                    sample_size = (128, 512),
                    pars=False
                    ):

    dataset = AdvDiffDataset(
            data_path=data_path,
            num_files=num_files,
            transformer_state=transformer_state,
            transformer_pars=transformer_pars,
            num_states_pr_sample=num_states_pr_sample,
            sample_size=(128, 512),
            pars=pars
    )
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last
    )

    return dataloader