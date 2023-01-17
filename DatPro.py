from nptdms import TdmsFile
from Fun.Fun_Nor import Fun_Nor
from Fun.DatSeg import DatSeg
import numpy as Num
from torch.utils.data import TensorDataset, DataLoader
import torch
def DatPro(Sec=5, Fre_Sam=1024, Rat_Tra=0.9, Win=1024,keywords='d_1000-3000'):

    C1212_N = TdmsFile('Dat/con_/con_12_n/'+keywords+'_n_con_12cm.tdms')
    C1212_F = TdmsFile('Dat/con_/con_12_f/'+keywords+'_f_con_12cm.tdms')
    C1214_N = TdmsFile('Dat/con_/con_14_n/'+keywords+'_n_con_14cm.tdms')
    C1214_F = TdmsFile('Dat/con_/con_14_f/'+keywords+'_f_con_14cm.tdms')

    I1212_N = TdmsFile('Dat/ips_/ips_12_n/'+keywords+'_n_ips_12cm.tdms')
    I1212_F = TdmsFile('Dat/ips_/ips_12_f/'+keywords+'_f_ips_12cm.tdms')
    I1214_N = TdmsFile('Dat/ips_/ips_14_n/'+keywords+'_n_ips_14cm.tdms')
    I1214_F = TdmsFile('Dat/ips_/ips_14_f/'+keywords+'_f_ips_14cm.tdms')

    V1212_N = TdmsFile('Dat/top_/top_12_n/'+keywords+'_n_top_12cm.tdms')
    V1212_F = TdmsFile('Dat/top_/top_12_f/'+keywords+'_f_top_12cm.tdms')
    V1214_N = TdmsFile('Dat/top_/top_14_n/'+keywords+'_n_top_14cm.tdms')
    V1214_F = TdmsFile('Dat/top_/top_14_f/'+keywords+'_f_top_14cm.tdms')


    #
    C1212_N0 = C1212_N['Untitled']['Sound Pressure'].data  # Sound Pressure_0  Voltage
    C1212_N1 = C1212_N['Untitled']['Sound Pressure_0'].data
    C1212_F0 = C1212_F['Untitled']['Sound Pressure'].data  # Sound Pressure_0  Voltage
    C1212_F1 = C1212_F['Untitled']['Sound Pressure_0'].data
    C1214_N0 = C1214_N['Untitled']['Sound Pressure'].data  # Sound Pressure_0  Voltage
    C1214_N1 = C1214_N['Untitled']['Sound Pressure_0'].data
    C1214_F0 = C1214_F['Untitled']['Sound Pressure'].data  # Sound Pressure_0  Voltage
    C1214_F1 = C1214_F['Untitled']['Sound Pressure_0'].data

    I1212_N0 = I1212_N['Untitled']['Sound Pressure'].data  # Sound Pressure_0  Voltage
    I1212_N1 = I1212_N['Untitled']['Sound Pressure_0'].data
    I1212_F0 = I1212_F['Untitled']['Sound Pressure'].data  # Sound Pressure_0  Voltage
    I1212_F1 = I1212_F['Untitled']['Sound Pressure_0'].data
    I1214_N0 = I1214_N['Untitled']['Sound Pressure'].data  # Sound Pressure_0  Voltage
    I1214_N1 = I1214_N['Untitled']['Sound Pressure_0'].data
    I1214_F0 = I1214_F['Untitled']['Sound Pressure'].data  # Sound Pressure_0  Voltage
    I1214_F1 = I1214_F['Untitled']['Sound Pressure_0'].data

    V1212_N0 = V1212_N['Untitled']['Sound Pressure'].data  # Sound Pressure_0  Voltage
    V1212_N1 = V1212_N['Untitled']['Sound Pressure_0'].data
    V1212_F0 = V1212_F['Untitled']['Sound Pressure'].data  # Sound Pressure_0  Voltage
    V1212_F1 = V1212_F['Untitled']['Sound Pressure_0'].data
    V1214_N0 = V1214_N['Untitled']['Sound Pressure'].data  # Sound Pressure_0  Voltage
    V1214_N1 = V1214_N['Untitled']['Sound Pressure_0'].data
    V1214_F0 = V1214_F['Untitled']['Sound Pressure'].data  # Sound Pressure_0  Voltage
    V1214_F1 = V1214_F['Untitled']['Sound Pressure_0'].data

    ##
    C12_N0 = C1212_N0[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    C12_N1 = C1212_N1[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    C12_F0 = C1212_F0[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    C12_F1 = C1212_F1[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    C14_N0 = C1214_N0[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    C14_N1 = C1214_N1[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    C14_F0 = C1214_F0[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    C14_F1 = C1214_F1[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]

    I12_N0 = I1212_N0[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    I12_N1 = I1212_N1[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    I12_F0 = I1212_F0[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    I12_F1 = I1212_F1[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    I14_N0 = I1214_N0[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    I14_N1 = I1214_N1[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    I14_F0 = I1214_F0[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    I14_F1 = I1214_F1[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]

    V12_N0 = V1212_N0[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    V12_N1 = V1212_N1[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    V12_F0 = V1212_F0[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    V12_F1 = V1212_F1[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    V14_N0 = V1214_N0[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    V14_N1 = V1214_N1[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    V14_F0 = V1214_F0[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]
    V14_F1 = V1214_F1[0:int(25.6e3 * Sec):int(25.6e3/Fre_Sam)]

    #
    C12_Nor_N0 = Fun_Nor(C12_N0, Mod='Abs')
    C12_Nor_N1 = Fun_Nor(C12_N1, Mod='Abs')
    C12_Nor_F0 = Fun_Nor(C12_F0, Mod='Abs')
    C12_Nor_F1 = Fun_Nor(C12_F1, Mod='Abs')
    C14_Nor_N0 = Fun_Nor(C14_N0, Mod='Abs')
    C14_Nor_N1 = Fun_Nor(C14_N1, Mod='Abs')
    C14_Nor_F0 = Fun_Nor(C14_F0, Mod='Abs')
    C14_Nor_F1 = Fun_Nor(C14_F1, Mod='Abs')

    I12_Nor_N0 = Fun_Nor(I12_N0, Mod='Abs')
    I12_Nor_N1 = Fun_Nor(I12_N1, Mod='Abs')
    I12_Nor_F0 = Fun_Nor(I12_F0, Mod='Abs')
    I12_Nor_F1 = Fun_Nor(I12_F1, Mod='Abs')
    I14_Nor_N0 = Fun_Nor(I14_N0, Mod='Abs')
    I14_Nor_N1 = Fun_Nor(I14_N1, Mod='Abs')
    I14_Nor_F0 = Fun_Nor(I14_F0, Mod='Abs')
    I14_Nor_F1 = Fun_Nor(I14_F1, Mod='Abs')

    V12_Nor_N0 = Fun_Nor(V12_N0, Mod='Abs')
    V12_Nor_N1 = Fun_Nor(V12_N1, Mod='Abs')
    V12_Nor_F0 = Fun_Nor(V12_F0, Mod='Abs')
    V12_Nor_F1 = Fun_Nor(V12_F1, Mod='Abs')
    V14_Nor_N0 = Fun_Nor(V14_N0, Mod='Abs')
    V14_Nor_N1 = Fun_Nor(V14_N1, Mod='Abs')
    V14_Nor_F0 = Fun_Nor(V14_F0, Mod='Abs')
    V14_Nor_F1 = Fun_Nor(V14_F1, Mod='Abs')

    C12_TraX_N0, C12_TraY_N0, C12_TesX_N0, C12_TesY_N0 = DatSeg(C12_Nor_N0, Win, Rat_Tra, Lab=0., Mod='Fre', Fre_Foc=[10,310])
    C12_TraX_F0, C12_TraY_F0, C12_TesX_F0, C12_TesY_F0 = DatSeg(C12_Nor_F0, Win, Rat_Tra, Lab=1., Mod='Fre', Fre_Foc=[10,310])
    C12_TraX_N1, C12_TraY_N1, C12_TesX_N1, C12_TesY_N1 = DatSeg(C12_Nor_N1, Win, Rat_Tra, Lab=0., Mod='Fre', Fre_Foc=[10,310])
    C12_TraX_F1, C12_TraY_F1, C12_TesX_F1, C12_TesY_F1 = DatSeg(C12_Nor_F1, Win, Rat_Tra, Lab=1., Mod='Fre', Fre_Foc=[10,310])
    C14_TraX_N0, C14_TraY_N0, C14_TesX_N0, C14_TesY_N0 = DatSeg(C14_Nor_N0, Win, Rat_Tra, Lab=0., Mod='Fre', Fre_Foc=[10,310])
    C14_TraX_F0, C14_TraY_F0, C14_TesX_F0, C14_TesY_F0 = DatSeg(C14_Nor_F0, Win, Rat_Tra, Lab=1., Mod='Fre', Fre_Foc=[10,310])
    C14_TraX_N1, C14_TraY_N1, C14_TesX_N1, C14_TesY_N1 = DatSeg(C14_Nor_N1, Win, Rat_Tra, Lab=0., Mod='Fre', Fre_Foc=[10,310])
    C14_TraX_F1, C14_TraY_F1, C14_TesX_F1, C14_TesY_F1 = DatSeg(C14_Nor_F1, Win, Rat_Tra, Lab=1., Mod='Fre', Fre_Foc=[10,310])

    I12_TraX_N0, I12_TraY_N0, I12_TesX_N0, I12_TesY_N0 = DatSeg(I12_Nor_N0, Win, Rat_Tra, Lab=0., Mod='Fre', Fre_Foc=[10,310])
    I12_TraX_F0, I12_TraY_F0, I12_TesX_F0, I12_TesY_F0 = DatSeg(I12_Nor_F0, Win, Rat_Tra, Lab=1., Mod='Fre', Fre_Foc=[10,310])
    I12_TraX_N1, I12_TraY_N1, I12_TesX_N1, I12_TesY_N1 = DatSeg(I12_Nor_N1, Win, Rat_Tra, Lab=0., Mod='Fre', Fre_Foc=[10,310])
    I12_TraX_F1, I12_TraY_F1, I12_TesX_F1, I12_TesY_F1 = DatSeg(I12_Nor_F1, Win, Rat_Tra, Lab=1., Mod='Fre', Fre_Foc=[10,310])
    I14_TraX_N0, I14_TraY_N0, I14_TesX_N0, I14_TesY_N0 = DatSeg(I14_Nor_N0, Win, Rat_Tra, Lab=0., Mod='Fre', Fre_Foc=[10,310])
    I14_TraX_F0, I14_TraY_F0, I14_TesX_F0, I14_TesY_F0 = DatSeg(I14_Nor_F0, Win, Rat_Tra, Lab=1., Mod='Fre', Fre_Foc=[10,310])
    I14_TraX_N1, I14_TraY_N1, I14_TesX_N1, I14_TesY_N1 = DatSeg(I14_Nor_N1, Win, Rat_Tra, Lab=0., Mod='Fre', Fre_Foc=[10,310])
    I14_TraX_F1, I14_TraY_F1, I14_TesX_F1, I14_TesY_F1 = DatSeg(I14_Nor_F1, Win, Rat_Tra, Lab=1., Mod='Fre', Fre_Foc=[10,310])

    V12_TraX_N0, V12_TraY_N0, V12_TesX_N0, V12_TesY_N0 = DatSeg(V12_Nor_N0, Win, Rat_Tra, Lab=0., Mod='Fre', Fre_Foc=[10,310])
    V12_TraX_F0, V12_TraY_F0, V12_TesX_F0, V12_TesY_F0 = DatSeg(V12_Nor_F0, Win, Rat_Tra, Lab=1., Mod='Fre', Fre_Foc=[10,310])
    V12_TraX_N1, V12_TraY_N1, V12_TesX_N1, V12_TesY_N1 = DatSeg(V12_Nor_N1, Win, Rat_Tra, Lab=0., Mod='Fre', Fre_Foc=[10,310])
    V12_TraX_F1, V12_TraY_F1, V12_TesX_F1, V12_TesY_F1 = DatSeg(V12_Nor_F1, Win, Rat_Tra, Lab=1., Mod='Fre', Fre_Foc=[10,310])
    V14_TraX_N0, V14_TraY_N0, V14_TesX_N0, V14_TesY_N0 = DatSeg(V14_Nor_N0, Win, Rat_Tra, Lab=0., Mod='Fre', Fre_Foc=[10,310])
    V14_TraX_F0, V14_TraY_F0, V14_TesX_F0, V14_TesY_F0 = DatSeg(V14_Nor_F0, Win, Rat_Tra, Lab=1., Mod='Fre', Fre_Foc=[10,310])
    V14_TraX_N1, V14_TraY_N1, V14_TesX_N1, V14_TesY_N1 = DatSeg(V14_Nor_N1, Win, Rat_Tra, Lab=0., Mod='Fre', Fre_Foc=[10,310])
    V14_TraX_F1, V14_TraY_F1, V14_TesX_F1, V14_TesY_F1 = DatSeg(V14_Nor_F1, Win, Rat_Tra, Lab=1., Mod='Fre', Fre_Foc=[10, 310])

    C12_TraX_0 = Num.concatenate((C12_TraX_N0, C12_TraX_F0), axis=0)
    C12_TraY_0 = Num.concatenate((C12_TraY_N0, C12_TraY_F0), axis=0)
    C12_TesX_0 = Num.concatenate((C12_TesX_N0, C12_TesX_F0), axis=0)
    C12_TesY_0 = Num.concatenate((C12_TesY_N0, C12_TesY_F0), axis=0)
    C12_TraX_1 = Num.concatenate((C12_TraX_N1, C12_TraX_F1), axis=0)
    C12_TraY_1 = Num.concatenate((C12_TraY_N1, C12_TraY_F1), axis=0)
    C12_TesX_1 = Num.concatenate((C12_TesX_N1, C12_TesX_F1), axis=0)
    C12_TesY_1 = Num.concatenate((C12_TesY_N1, C12_TesY_F1), axis=0)
    C14_TraX_0 = Num.concatenate((C14_TraX_N0, C14_TraX_F0), axis=0)
    C14_TraY_0 = Num.concatenate((C14_TraY_N0, C14_TraY_F0), axis=0)
    C14_TesX_0 = Num.concatenate((C14_TesX_N0, C14_TesX_F0), axis=0)
    C14_TesY_0 = Num.concatenate((C14_TesY_N0, C14_TesY_F0), axis=0)
    C14_TraX_1 = Num.concatenate((C14_TraX_N1, C14_TraX_F1), axis=0)
    C14_TraY_1 = Num.concatenate((C14_TraY_N1, C14_TraY_F1), axis=0)
    C14_TesX_1 = Num.concatenate((C14_TesX_N1, C14_TesX_F1), axis=0)
    C14_TesY_1 = Num.concatenate((C14_TesY_N1, C14_TesY_F1), axis=0)

    I12_TraX_0 = Num.concatenate((I12_TraX_N0, I12_TraX_F0), axis=0)
    I12_TraY_0 = Num.concatenate((I12_TraY_N0, I12_TraY_F0), axis=0)
    I12_TesX_0 = Num.concatenate((I12_TesX_N0, I12_TesX_F0), axis=0)
    I12_TesY_0 = Num.concatenate((I12_TesY_N0, I12_TesY_F0), axis=0)
    I12_TraX_1 = Num.concatenate((I12_TraX_N1, I12_TraX_F1), axis=0)
    I12_TraY_1 = Num.concatenate((I12_TraY_N1, I12_TraY_F1), axis=0)
    I12_TesX_1 = Num.concatenate((I12_TesX_N1, I12_TesX_F1), axis=0)
    I12_TesY_1 = Num.concatenate((I12_TesY_N1, I12_TesY_F1), axis=0)
    I14_TraX_0 = Num.concatenate((I14_TraX_N0, I14_TraX_F0), axis=0)
    I14_TraY_0 = Num.concatenate((I14_TraY_N0, I14_TraY_F0), axis=0)
    I14_TesX_0 = Num.concatenate((I14_TesX_N0, I14_TesX_F0), axis=0)
    I14_TesY_0 = Num.concatenate((I14_TesY_N0, I14_TesY_F0), axis=0)
    I14_TraX_1 = Num.concatenate((I14_TraX_N1, I14_TraX_F1), axis=0)
    I14_TraY_1 = Num.concatenate((I14_TraY_N1, I14_TraY_F1), axis=0)
    I14_TesX_1 = Num.concatenate((I14_TesX_N1, I14_TesX_F1), axis=0)
    I14_TesY_1 = Num.concatenate((I14_TesY_N1, I14_TesY_F1), axis=0)

    V12_TraX_0 = Num.concatenate((V12_TraX_N0, V12_TraX_F0), axis=0)
    V12_TraY_0 = Num.concatenate((V12_TraY_N0, V12_TraY_F0), axis=0)
    V12_TesX_0 = Num.concatenate((V12_TesX_N0, V12_TesX_F0), axis=0)
    V12_TesY_0 = Num.concatenate((V12_TesY_N0, V12_TesY_F0), axis=0)
    V12_TraX_1 = Num.concatenate((V12_TraX_N1, V12_TraX_F1), axis=0)
    V12_TraY_1 = Num.concatenate((V12_TraY_N1, V12_TraY_F1), axis=0)
    V12_TesX_1 = Num.concatenate((V12_TesX_N1, V12_TesX_F1), axis=0)
    V12_TesY_1 = Num.concatenate((V12_TesY_N1, V12_TesY_F1), axis=0)
    V14_TraX_0 = Num.concatenate((V14_TraX_N0, V14_TraX_F0), axis=0)
    V14_TraY_0 = Num.concatenate((V14_TraY_N0, V14_TraY_F0), axis=0)
    V14_TesX_0 = Num.concatenate((V14_TesX_N0, V14_TesX_F0), axis=0)
    V14_TesY_0 = Num.concatenate((V14_TesY_N0, V14_TesY_F0), axis=0)
    V14_TraX_1 = Num.concatenate((V14_TraX_N1, V14_TraX_F1), axis=0)
    V14_TraY_1 = Num.concatenate((V14_TraY_N1, V14_TraY_F1), axis=0)
    V14_TesX_1 = Num.concatenate((V14_TesX_N1, V14_TesX_F1), axis=0)
    V14_TesY_1 = Num.concatenate((V14_TesY_N1, V14_TesY_F1), axis=0)

    Tra_Pos1 = TensorDataset(torch.tensor(C12_TraX_1).to(torch.float32), torch.tensor(C12_TraY_1).to(torch.long))
    Tes_Pos1 = TensorDataset(torch.tensor(C12_TesX_1).to(torch.float32), torch.tensor(C12_TesY_1).to(torch.long))
    Tra_Pos2 = TensorDataset(torch.tensor(V12_TraX_1).to(torch.float32), torch.tensor(V12_TraY_1).to(torch.long))  # 14-12
    Tes_Pos2 = TensorDataset(torch.tensor(V12_TesX_1).to(torch.float32), torch.tensor(V12_TesY_1).to(torch.long))
    Tra_Pos3 = TensorDataset(torch.tensor(C12_TraX_0).to(torch.float32), torch.tensor(C12_TraY_0).to(torch.long))
    Tes_Pos3 = TensorDataset(torch.tensor(C12_TesX_0).to(torch.float32), torch.tensor(C12_TesY_0).to(torch.long))




    T12_TraX_N = Num.concatenate((C12_TraX_N0, C12_TraX_N1), axis=1)
    T12_TraY_N = C12_TraY_N0
    T12_TraX_F = Num.concatenate((C12_TraX_F0, C12_TraX_F1), axis=1)
    T12_TraY_F = C12_TraY_F0
    T12_TesX_N = Num.concatenate((C12_TesX_N0, C12_TesX_N1), axis=1)
    T12_TesY_N = C12_TesY_N0
    T12_TesX_F = Num.concatenate((C12_TesX_F0, C12_TesX_F1), axis=1)
    T12_TesY_F = C12_TesY_F0
    T14_TraX_N = Num.concatenate((C14_TraX_N0, C14_TraX_N1), axis=1)
    T14_TraY_N = C14_TraY_N0
    T14_TraX_F = Num.concatenate((C14_TraX_F0, C14_TraX_F1), axis=1)
    T14_TraY_F = C14_TraY_F0
    T14_TesX_N = Num.concatenate((C14_TesX_N0, C14_TesX_N1), axis=1)
    T14_TesY_N = C14_TesY_N0
    T14_TesX_F = Num.concatenate((C14_TesX_F0, C14_TesX_F1), axis=1)
    T14_TesY_F = C14_TesY_F0
    C12_TraX = Num.concatenate((T12_TraX_N, T12_TraX_F), axis=0)
    C12_TraY = Num.concatenate((T12_TraY_N, T12_TraY_F), axis=0)
    C12_TesX = Num.concatenate((T12_TesX_N, T12_TesX_F), axis=0)
    C12_TesY = Num.concatenate((T12_TesY_N, T12_TesY_F), axis=0)
    C14_TraX = Num.concatenate((T14_TraX_N, T14_TraX_F), axis=0)
    C14_TraY = Num.concatenate((T14_TraY_N, T14_TraY_F), axis=0)
    C14_TesX = Num.concatenate((T14_TesX_N, T14_TesX_F), axis=0)
    C14_TesY = Num.concatenate((T14_TesY_N, T14_TesY_F), axis=0)

    C12_TraX_E = Num.concatenate((C12_TraX_N0-C12_TraX_N1, C12_TraX_F0-C12_TraX_F1), axis=0)
    C12_TraY_E = Num.concatenate((C12_TraY_N0, C12_TraY_F0))
    C12_TesX_E = Num.concatenate((C12_TesX_N0-C12_TesX_N1, C12_TesX_F0-C12_TesX_F1), axis=0)
    C12_TesY_E = Num.concatenate((C12_TesY_N0, C12_TesY_F0))
    C14_TraX_E = Num.concatenate((C14_TraX_N0-C14_TraX_N1, C14_TraX_F0-C14_TraX_F1), axis=0)
    C14_TraY_E = Num.concatenate((C14_TraY_N0, C14_TraY_F0))
    C14_TesX_E = Num.concatenate((C14_TesX_N0-C14_TesX_N1, C14_TesX_F0-C14_TesX_F1), axis=0)
    C14_TesY_E = Num.concatenate((C14_TesY_N0, C14_TesY_F0))




    T12_TraX_N = Num.concatenate((I12_TraX_N0, I12_TraX_N1), axis=1)
    T12_TraY_N = I12_TraY_N0
    T12_TraX_F = Num.concatenate((I12_TraX_F0, I12_TraX_F1), axis=1)
    T12_TraY_F = I12_TraY_F0
    T12_TesX_N = Num.concatenate((I12_TesX_N0, I12_TesX_N1), axis=1)
    T12_TesY_N = I12_TesY_N0
    T12_TesX_F = Num.concatenate((I12_TesX_F0, I12_TesX_F1), axis=1)
    T12_TesY_F = I12_TesY_F0
    T14_TraX_N = Num.concatenate((I14_TraX_N0, I14_TraX_N1), axis=1)
    T14_TraY_N = I14_TraY_N0
    T14_TraX_F = Num.concatenate((I14_TraX_F0, I14_TraX_F1), axis=1)
    T14_TraY_F = I14_TraY_F0
    T14_TesX_N = Num.concatenate((I14_TesX_N0, I14_TesX_N1), axis=1)
    T14_TesY_N = I14_TesY_N0
    T14_TesX_F = Num.concatenate((I14_TesX_F0, I14_TesX_F1), axis=1)
    T14_TesY_F = I14_TesY_F0
    I12_TraX = Num.concatenate((T12_TraX_N, T12_TraX_F), axis=0)
    I12_TraY = Num.concatenate((T12_TraY_N, T12_TraY_F), axis=0)
    I12_TesX = Num.concatenate((T12_TesX_N, T12_TesX_F), axis=0)
    I12_TesY = Num.concatenate((T12_TesY_N, T12_TesY_F), axis=0)
    I14_TraX = Num.concatenate((T14_TraX_N, T14_TraX_F), axis=0)
    I14_TraY = Num.concatenate((T14_TraY_N, T14_TraY_F), axis=0)
    I14_TesX = Num.concatenate((T14_TesX_N, T14_TesX_F), axis=0)
    I14_TesY = Num.concatenate((T14_TesY_N, T14_TesY_F), axis=0)
    I12_TraX_E = Num.concatenate((I12_TraX_N0-I12_TraX_N1, I12_TraX_F0-I12_TraX_F1), axis=0)
    I12_TraY_E = Num.concatenate((I12_TraY_N0, I12_TraY_F0))
    I12_TesX_E = Num.concatenate((I12_TesX_N0-I12_TesX_N1, I12_TesX_F0-I12_TesX_F1), axis=0)
    I12_TesY_E = Num.concatenate((I12_TesY_N0, I12_TesY_F0))
    I14_TraX_E = Num.concatenate((I14_TraX_N0-I14_TraX_N1, I14_TraX_F0-I14_TraX_F1), axis=0)
    I14_TraY_E = Num.concatenate((I14_TraY_N0, I14_TraY_F0))
    I14_TesX_E = Num.concatenate((I14_TesX_N0-I14_TesX_N1, I14_TesX_F0-I14_TesX_F1), axis=0)
    I14_TesY_E = Num.concatenate((I14_TesY_N0, I14_TesY_F0))


    T12_TraX_N = Num.concatenate((V12_TraX_N0, V12_TraX_N1), axis=1)
    T12_TraY_N = V12_TraY_N0
    T12_TraX_F = Num.concatenate((V12_TraX_F0, V12_TraX_F1), axis=1)
    T12_TraY_F = V12_TraY_F0
    T12_TesX_N = Num.concatenate((V12_TesX_N0, V12_TesX_N1), axis=1)
    T12_TesY_N = V12_TesY_N0
    T12_TesX_F = Num.concatenate((V12_TesX_F0, V12_TesX_F1), axis=1)
    T12_TesY_F = V12_TesY_F0
    T14_TraX_N = Num.concatenate((V14_TraX_N0, V14_TraX_N1), axis=1)
    T14_TraY_N = V14_TraY_N0
    T14_TraX_F = Num.concatenate((V14_TraX_F0, V14_TraX_F1), axis=1)
    T14_TraY_F = V14_TraY_F0
    T14_TesX_N = Num.concatenate((V14_TesX_N0, V14_TesX_N1), axis=1)
    T14_TesY_N = V14_TesY_N0
    T14_TesX_F = Num.concatenate((V14_TesX_F0, V14_TesX_F1), axis=1)
    T14_TesY_F = V14_TesY_F0
    V12_TraX = Num.concatenate((T12_TraX_N, T12_TraX_F), axis=0)
    V12_TraY = Num.concatenate((T12_TraY_N, T12_TraY_F), axis=0)
    V12_TesX = Num.concatenate((T12_TesX_N, T12_TesX_F), axis=0)
    V12_TesY = Num.concatenate((T12_TesY_N, T12_TesY_F), axis=0)
    V14_TraX = Num.concatenate((T14_TraX_N, T14_TraX_F), axis=0)
    V14_TraY = Num.concatenate((T14_TraY_N, T14_TraY_F), axis=0)
    V14_TesX = Num.concatenate((T14_TesX_N, T14_TesX_F), axis=0)
    V14_TesY = Num.concatenate((T14_TesY_N, T14_TesY_F), axis=0)
    V12_TraX_E = Num.concatenate((V12_TraX_N0-V12_TraX_N1, V12_TraX_F0-V12_TraX_F1), axis=0)
    V12_TraY_E = Num.concatenate((V12_TraY_N0, V12_TraY_F0))
    V12_TesX_E = Num.concatenate((V12_TesX_N0-V12_TesX_N1, V12_TesX_F0-V12_TesX_F1), axis=0)
    V12_TesY_E = Num.concatenate((V12_TesY_N0, V12_TesY_F0))
    V14_TraX_E = Num.concatenate((V14_TraX_N0-V14_TraX_N1, V14_TraX_F0-V14_TraX_F1), axis=0)
    V14_TraY_E = Num.concatenate((V14_TraY_N0, V14_TraY_F0))
    V14_TesX_E = Num.concatenate((V14_TesX_N0-V14_TesX_N1, V14_TesX_F0-V14_TesX_F1), axis=0)
    V14_TesY_E = Num.concatenate((V14_TesY_N0, V14_TesY_F0))

    Tra_Pos4 = TensorDataset(torch.tensor(C12_TraX).to(torch.float32), torch.tensor(C12_TraY).to(torch.long))
    Tes_Pos4 = TensorDataset(torch.tensor(C12_TesX).to(torch.float32), torch.tensor(C12_TesY).to(torch.long))

    Tra_Pos5 = TensorDataset(torch.tensor(V12_TraX).to(torch.float32), torch.tensor(V12_TraY).to(torch.long))
    Tes_Pos5 = TensorDataset(torch.tensor(V12_TesX).to(torch.float32), torch.tensor(V12_TesY).to(torch.long))

    Tra_Pos6 = TensorDataset(torch.tensor(I12_TraX).to(torch.float32), torch.tensor(I12_TraY).to(torch.long))
    Tes_Pos6 = TensorDataset(torch.tensor(I12_TesX).to(torch.float32), torch.tensor(I12_TesY).to(torch.long))

    Tra_Pos7 = TensorDataset(torch.tensor(C14_TraX).to(torch.float32), torch.tensor(C14_TraY).to(torch.long))
    Tes_Pos7 = TensorDataset(torch.tensor(C14_TesX).to(torch.float32), torch.tensor(C14_TesY).to(torch.long))

    Tra_Pos8 = TensorDataset(torch.tensor(V14_TraX).to(torch.float32), torch.tensor(V14_TraY).to(torch.long))
    Tes_Pos8 = TensorDataset(torch.tensor(V14_TesX).to(torch.float32), torch.tensor(V14_TesY).to(torch.long))

    Tra_Pos9 = TensorDataset(torch.tensor(I14_TraX).to(torch.float32), torch.tensor(I14_TraY).to(torch.long))
    Tes_Pos9 = TensorDataset(torch.tensor(I14_TesX).to(torch.float32), torch.tensor(I14_TesY).to(torch.long))

    Tra_Pos4_E = TensorDataset(torch.tensor(C12_TraX_E).to(torch.float32), torch.tensor(C12_TraY_E).to(torch.long))
    Tes_Pos4_E = TensorDataset(torch.tensor(C12_TesX_E).to(torch.float32), torch.tensor(C12_TesY_E).to(torch.long))

    Tra_Pos5_E = TensorDataset(torch.tensor(V12_TraX_E).to(torch.float32), torch.tensor(V12_TraY_E).to(torch.long))
    Tes_Pos5_E = TensorDataset(torch.tensor(V12_TesX_E).to(torch.float32), torch.tensor(V12_TesY_E).to(torch.long))

    Tra_Pos6_E = TensorDataset(torch.tensor(I12_TraX_E).to(torch.float32), torch.tensor(I12_TraY_E).to(torch.long))
    Tes_Pos6_E = TensorDataset(torch.tensor(I12_TesX_E).to(torch.float32), torch.tensor(I12_TesY_E).to(torch.long))

    Tra_Pos7_E = TensorDataset(torch.tensor(C14_TraX_E).to(torch.float32), torch.tensor(C14_TraY_E).to(torch.long))
    Tes_Pos7_E = TensorDataset(torch.tensor(C14_TesX_E).to(torch.float32), torch.tensor(C14_TesY_E).to(torch.long))

    Tra_Pos8_E = TensorDataset(torch.tensor(V14_TraX_E).to(torch.float32), torch.tensor(V14_TraY_E).to(torch.long))
    Tes_Pos8_E = TensorDataset(torch.tensor(V14_TesX_E).to(torch.float32), torch.tensor(V14_TesY_E).to(torch.long))

    Tra_Pos9_E = TensorDataset(torch.tensor(I14_TraX_E).to(torch.float32), torch.tensor(I14_TraY_E).to(torch.long))
    Tes_Pos9_E = TensorDataset(torch.tensor(I14_TesX_E).to(torch.float32), torch.tensor(I14_TesY_E).to(torch.long))

    Out = [Tra_Pos1, Tes_Pos1, Tra_Pos2, Tes_Pos2, Tra_Pos3, Tes_Pos3, Tra_Pos4, Tes_Pos4, Tra_Pos5, Tes_Pos5,  # 0-6
            Tra_Pos6, Tes_Pos6, Tra_Pos7, Tes_Pos7, Tra_Pos8, Tes_Pos8, Tra_Pos9, Tes_Pos9, Tra_Pos4_E, Tes_Pos4_E,  # 7-18
            Tra_Pos5_E, Tes_Pos5_E, Tra_Pos6_E, Tes_Pos6_E, Tra_Pos7_E, Tes_Pos7_E, Tra_Pos8_E, Tes_Pos8_E, Tra_Pos9_E, Tes_Pos9_E]  # 19-30

    return Out

if __name__ == '__main__':
    DatPro()