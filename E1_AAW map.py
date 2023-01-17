import numpy as np
import snntorch as snn
import torch
import numpy as Num
import math as Mat
import torch.nn as nn
import matplotlib.pyplot as Plt
from snntorch import spikeplot as splt
from snntorch import spikegen
import os
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from torch.utils.data import TensorDataset, DataLoader

from matplotlib import pyplot as plt
from DatPro import DatPro


def seed_torch(seed=1):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()
#

# DatSet = DatPro(Sec=10, Rat_Tra=0.8, keywords=keywords)
#
# Tra = []
# Tes = []
# batch_size = 128
# for i in range(6):
#     Tra.append(DataLoader(DatSet[i*2], batch_size=batch_size, shuffle=True, drop_last=True))
#     Tes.append(DataLoader(DatSet[i*2+1], batch_size=batch_size, shuffle=True, drop_last=True))

################################################

keywords = 's_1000' # s_1000 s_2000 s_3000 d_1000-3000 map for------ default s_1000
tranum = 1 # optional 0 1 2 ... default 0

for i in range(6):
    StrCom = f"Net_{i:d}"+" = torch.load('Net/E0/"+keywords+f"/Net_{i}_{tranum:d}.pkl')"
    exec(StrCom)

for i in range(3):
    for j in range(2):
        StrCom = f'W_{i:d}_{j:d}_0=Net_{i:d}.fc{j+1}_0.weight.detach().numpy()'
        exec(StrCom)

for i in range(3,6):
    StrCom = f'W_{i:d}_0_0=Net_{i:d}.fc1_0.weight.detach().numpy()'
    exec(StrCom)
    StrCom = f'W_{i:d}_0_1=Net_{i:d}.fc1_1.weight.detach().numpy()'
    exec(StrCom)
    StrCom = f'W_{i:d}_1_0=Net_{i:d}.fc2_0.weight.detach().numpy()'
    exec(StrCom)

# axe1 = fig.add_subplot(3,2,1)
# # axe1.set_title('Single sensor: net_0_0')
# axe1.set_xlabel('Index of neurons', fontsize=8)
# axe1.set_ylabel('Average of absolute weights', fontsize=8)
# axe1.set_xlim(-10,300)
# axe1.set_ylim(0,0.15)
# axe1.plot(W00)

## AAW of six positions
fig0 = plt.figure(figsize=(6,8))
fontsize = 8
for i in range(3):
    for j in range(2):
        axe = fig0.add_subplot(3, 2, i*2+j+1)
        StrCom = f"W = Num.mean(Num.transpose(abs(W_{i:d}_{j:d}_0)), axis=1)"
        exec(StrCom)
        axe.plot(W)
        axe.set_xlabel('Neuron index',fontsize=fontsize)
        axe.set_ylabel('AAW',fontsize=fontsize)
        axe.set_ylim([0,1])
        if j==1:
            axe.set_title(f'AAW_{i:d}_{j:d}_0',color='red')
        else:
            axe.set_title(f'AAW_{i:d}_{j:d}_0')

fig0.subplots_adjust(wspace=0.4,hspace=0.4,top=0.95,right=0.95,left=0.12,bottom=0.07)
filename = 'Fig/E1/'+keywords+'AAW single.pdf'
fig0.savefig(filename)

fig1 = plt.figure(figsize=(9,8))
fontsize = 8
for i in range(3,6):
    axe = fig1.add_subplot(3, 3, (i-3) * 3 + 1)
    StrCom = f"W = Num.mean(Num.transpose(abs(W_{i:d}_0_0)), axis=1)"
    exec(StrCom)
    axe.plot(W)
    axe.set_xlabel('Neuron index',fontsize=fontsize)
    axe.set_ylabel('AAW',fontsize=fontsize)
    axe.set_title(f'AAW_{i:d}_0_0')
    axe.set_ylim([0,1])

    axe = fig1.add_subplot(3, 3,  (i-3) * 3 + 2)
    StrCom = f"W = Num.mean(Num.transpose(abs(W_{i:d}_0_1)), axis=1)"
    exec(StrCom)
    axe.plot(W)
    axe.set_xlabel('Neuron index',fontsize=fontsize)
    axe.set_ylabel('AAW',fontsize=fontsize)
    axe.set_title(f'AAW_{i:d}_0_1')
    axe.set_ylim([0, 1])

    axe = fig1.add_subplot(3, 3,  (i-3) * 3 + 3)
    StrCom = f"W = Num.mean(Num.transpose(abs(W_{i:d}_1_0)), axis=1)"
    exec(StrCom)
    axe.plot(W)
    axe.set_xlabel('Neuron index',fontsize=fontsize)
    axe.set_ylabel('AAW',fontsize=fontsize)
    axe.set_title(f'AAW_{i:d}_1_0', color='red')
    axe.set_ylim([0, 1])

fig1.subplots_adjust(wspace=0.4,hspace=0.4,top=0.95,right=0.95,left=0.07,bottom=0.07)
filename = 'Fig/E1/'+keywords+'AAW double.pdf'
fig1.savefig(filename)

fig1.show()
# fig0.show()
a=10
b=10






#  AAW and spikes
#  net1
# fig2 = plt.figure(figsize=(6,8))
# fontsize = 8
# axe_0 = fig2.add_subplot(2, 2, 1)
# StrCom = f"W = Num.mean(Num.transpose(abs(W_1_0_0)), axis=1)"
# exec(StrCom)
# axe_0.plot(W)
# axe_0.set_xlabel('Neuron index', fontsize=fontsize)
# axe_0.set_ylabel('AAW', fontsize=fontsize)
#
# axe_1 = fig2.add_subplot(2, 2, 2)
# StrCom = f"W = Num.mean(Num.transpose(abs(W_1_1_0)), axis=1)"
# exec(StrCom)
# axe_1.plot(W)
# axe_1.set_xlabel('Neuron index', fontsize=fontsize)
# axe_1.set_ylabel('AAW', fontsize=fontsize)
#
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# with torch.no_grad():
#     net= Net_1_0 # Net_0_0
#     net.eval()
#     test_data, test_targets = next(iter(Tes[1]))
#     test_data = test_data.to(device)
#     test_targets = test_targets.to(device)
#     # Test set forward pass
#     spikes, = net(test_data.view(batch_size, -1))
#
# spk_his_1_0 = net.spike1_0_rec
# splt.raster(torch.tensor(spk_his_2), axe2, s=1.5, c="black")
#
# for i in range(1):
#     for j in range(2):
#         axe = fig2.add_subplot(2, 2, i*2+j+1)
#         StrCom = f"W = Num.mean(Num.transpose(abs(W_{i:d}_{j:d}_0)), axis=1)"
#         exec(StrCom)
#         axe.plot(W)
#         axe.set_xlabel('Neuron index',fontsize=fontsize)
#         axe.set_ylabel('AAW',fontsize=fontsize)
#         if j==1:
#             axe.set_title(f'W_{i:d}_{j:d}_0',color='red')
#         else:
#             axe.set_title(f'W_{i:d}_{j:d}_0')
#
# fig2.subplots_adjust(wspace=0.4,hspace=0.4,top=0.95,right=0.95,left=0.12,bottom=0.07)
# filename = 'Fig/E0/'+keywords+'AAW single.pdf'
# fig2.savefig(filename)
