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

######### attention use E0_Net 1000 rpm
########## draw weights and ratermap
def seed_torch(seed=10):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()

keywords = 's_1000' # s_1000 s_2000 s_3000 d_1000-3000 default: s_1000
tra_num = 1

DatSet = DatPro(Sec=10, Rat_Tra=0.8, keywords=keywords)
Tra = []
Tes = []
batch_size = 128
for i in range(6):
    Tra.append(DataLoader(DatSet[i*2], batch_size=batch_size, shuffle=True, drop_last=True))
    Tes.append(DataLoader(DatSet[i*2+1], batch_size=batch_size, shuffle=True, drop_last=True))
################################################

for i in range(6):
    StrCom = f"Net_{i:d}"+" = torch.load('Net/E0/"+keywords+f"/Net_{i}_{tra_num}.pkl')"
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

keywords0 = '1'  # 1
keywords1 = '5'  # 5
#  AAW and spikes
#  net1 W 1_0
fig2 = plt.figure(figsize=(6,6))
fontsize = 8
axe_0 = fig2.add_subplot(2, 2, 1)
StrCom = 'W = Num.mean(Num.transpose(abs(W_'+keywords0+'_1_0)), axis=1)'
exec(StrCom)
axe_0.plot(W)
axe_0.set_xlabel('Neuron index', fontsize=fontsize)
axe_0.set_ylabel('AAW', fontsize=fontsize)
axe_0.set_title('AAW_'+keywords0+'_1_0', color='red')
axe_0.set_xlim([0,200])
axe_0.set_ylim([0,1])


axe_1 = fig2.add_subplot(2, 2, 2)
StrCom = 'W = Num.mean(Num.transpose(abs(W_' + keywords1+ '_1_0)), axis=1)'
exec(StrCom)
axe_1.plot(W)
axe_1.set_xlabel('Neuron index', fontsize=fontsize)
axe_1.set_ylabel('AAW', fontsize=fontsize)
axe_1.set_title('AAW_'+keywords1+'_1_0', color='red')
axe_1.set_xlim([0,200])
axe_1.set_ylim([0,1])
# fig2.show()



######### net 3 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
with torch.no_grad():
    # net= Net_1_0 # Net_0_0
    StrCom = 'net=Net_'+keywords0
    exec(StrCom)
    net.spike1_0_rec = []
    net.spike1_1_rec = []
    net.eval()
    StrCom = 'test_data, test_targets = next(iter(Tes['+keywords0+']))'
    exec(StrCom)
    # test_data, test_targets = next(iter(Tes[1]))
    test_data = test_data.to(device)
    test_targets = test_targets.to(device)
    # Test set forward pass
    test_spikes,test_mem = net(test_data.view(batch_size, -1))

spk_his = []
for i in range(100):
    a = net.spike1_0_rec[i][0,:].numpy()
    # b = net.spike1_1_rec[i][0,:].numpy()
    # c = np.concatenate((a,b),axis=0)
    spk_his.append(a)
axe_2 = fig2.add_subplot(2, 2, 3)
spk_his_0 = np.array(spk_his)
spk_his_1 = np.round(spk_his_0).astype(int)
spk_his_2 = np.transpose(spk_his_1)
splt.raster(torch.tensor(spk_his_2), axe_2, s=1.5, c="black")
#spk_his_2
axe_2.set_ylim([0,100])
axe_2.set_title('R_'+keywords0+'_1_0', color='red')
axe_2.set_xlim([0,200])
axe_2.set_ylim([0,100])
axe_2.set_ylabel('Time step', fontsize=fontsize)
axe_2.set_xlabel('Neuron index', fontsize=fontsize)


################# net 5 0
with torch.no_grad():
    # net= Net_5_0 # Net_0_0
    StrCom = 'net=Net_'+keywords1
    exec(StrCom)
    net.spike1_0_rec = []
    net.spike1_1_rec = []
    net.eval()
    # test_data, test_targets = next(iter(Tes[5]))
    test_data = test_data.to(device)
    StrCom = 'test_data, test_targets = next(iter(Tes['+keywords1+']))'
    exec(StrCom)
    test_targets = test_targets.to(device)
    # Test set forward pass
    test_spikes,test_mem = net(test_data.view(batch_size, -1))
# time batch neuron
spk_his = []
for i in range(100):
    a = net.spike1_0_rec[i][0,:].numpy()
    b = net.spike1_1_rec[i][0,:].numpy()
    c = np.concatenate((a,b),axis=0)
    spk_his.append(c)

# fig2 = plt.figure(figsize=(6,8))
axe_3 = fig2.add_subplot(2, 2, 4)
spk_his_0 = np.array(spk_his)
spk_his_1 = np.round(spk_his_0).astype(int)
spk_his_2 = np.transpose(spk_his_1)
splt.raster(torch.tensor(spk_his_2), axe_3, s=1.5, c="black")
# d = np.sum(spk_his_2,axis=1)
axe_3.set_ylim([0,100])
axe_3.set_title('R_'+keywords1+'_1_0', color='red')
axe_3.set_xlim([0,200])
axe_3.set_ylim([0,100])
axe_3.set_ylabel('Time step', fontsize=fontsize)
axe_3.set_xlabel('Neuron index', fontsize=fontsize)

fig2.subplots_adjust(wspace=0.4,hspace=0.4,top=0.95,right=0.95,left=0.12,bottom=0.07)
fig2.show()
filename = 'Fig/E1/'+'Net'+keywords0+'Net'+keywords1+'WR map.pdf'
fig2.savefig(filename)

plt.figure()
a = np.mean(np.transpose(abs(W_1_1_0)), axis=1)
plt.plot(a)
plt.show()



