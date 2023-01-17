import numpy as np
import snntorch as snn
import torch
import numpy as Num
import math as Mat
import torch.nn as nn


class Net(nn.Module):
    def __init__(self,num_inputs,num_hidden,num_outputs,beta,num_steps,FlaSen):
        super().__init__()
        # Initialize layers
        self.num_steps = num_steps
        self.spike1_0_rec = []
        self.spike1_1_rec = []
        if FlaSen == 1:
            self.fc1_0= nn.Linear(num_inputs, num_hidden)
            self.lif1_0 = snn.Leaky(beta=beta)
            # self.drop = nn.Dropout(0.2)
            self.fc2_0 = nn.Linear(num_hidden, num_outputs)
            self.lif2_0 = snn.Leaky(beta=beta)
        else:
            self.fc1_0 = nn.Linear(int(num_inputs/2), int(num_hidden/2))
            self.lif1_0 = snn.Leaky(beta=beta)
            self.fc1_1 = nn.Linear(int(num_inputs/2), int(num_hidden/2))
            self.lif1_1 = snn.Leaky(beta=beta)

            # self.drop = nn.Dropout(0.2)
            self.fc2_0 = nn.Linear(num_hidden, num_outputs)
            self.lif2_0 = snn.Leaky(beta=beta)

    def forward(self, x):
        # Initialize hidden states at t=0
        if x.shape[1] == 300:
            mem1_0 = self.lif1_0.init_leaky()
            mem2_0 = self.lif2_0.init_leaky()
            # Record the final layer
            spk2_0_rec = []
            mem2_0_rec = []
            self.spike1_0_rec = []
            for step in range(self.num_steps):
                cur1_0 = self.fc1_0(x)
                spk1_0, mem1_0 = self.lif1_0(cur1_0, mem1_0)
                # Tem = self.drop(spk1)
                cur2_0 = self.fc2_0(spk1_0)
                spk2_0, mem2_0 = self.lif2_0(cur2_0, mem2_0)
                spk2_0_rec.append(spk2_0)
                mem2_0_rec.append(mem2_0)
                self.spike1_0_rec.append(spk1_0)
        else:
            mem1_0 = self.lif1_0.init_leaky()
            mem1_1 = self.lif1_1.init_leaky()
            mem2_0 = self.lif2_0.init_leaky()

            spk2_0_rec = []
            mem2_0_rec = []
            self.spike1_0_rec =[]
            self.spike1_1_rec =[]

            for step in range(self.num_steps):
                cur1_0 = self.fc1_0(x[:,0:300])
                spk1_0, mem1_0 = self.lif1_0(cur1_0, mem1_0)
                cur1_1 = self.fc1_1(x[:,300:600])
                spk1_1, mem1_1 = self.lif1_0(cur1_1, mem1_1)

                # Tem = self.drop(spk1)
                cur2_0 = self.fc2_0(torch.cat([spk1_0,spk1_1],1))
                spk2_0, mem2_0 = self.lif2_0(cur2_0, mem2_0)
                spk2_0_rec.append(spk2_0)
                mem2_0_rec.append(mem2_0)
                self.spike1_0_rec.append(spk1_0)
                self.spike1_1_rec.append(spk1_1)

        return torch.stack(spk2_0_rec, dim=0), torch.stack(mem2_0_rec, dim=0)