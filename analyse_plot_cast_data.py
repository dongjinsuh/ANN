import torch
import torchvision 
from torch.utils import data
from torchvision import transforms 
from torch import nn 
import numpy as np 
import math
import h5py 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

hf = h5py.File('../Dongjin/Cast_IAXO_data/calibration-cdl-2018.h5')


g1 = hf.get('calibration-cdl-feb2019-Ag-Ag-6kV/eccentricity')
g2 = hf.get('calibration-cdl-feb2019-Ag-Ag-6kV/energyFromCharge')
g3 = hf.get('calibration-cdl-feb2019-Ag-Ag-6kV/kurtosisLongitudinal')
g4 = hf.get('calibration-cdl-feb2019-Ag-Ag-6kV/kurtosisTransverse')
g5 = hf.get('calibration-cdl-feb2019-Ag-Ag-6kV/length')
g6 = hf.get('calibration-cdl-feb2019-Ag-Ag-6kV/skewnessLongitudinal')
g7 = hf.get('calibration-cdl-feb2019-Ag-Ag-6kV/skewnessTransverse')
g8 = hf.get('calibration-cdl-feb2019-Ag-Ag-6kV/fractionInTransverseRms')
g9 = hf.get('calibration-cdl-feb2019-Ag-Ag-6kV/hits')
g10 = hf.get('calibration-cdl-feb2019-Ag-Ag-6kV/rmsLongitudinal')
g11 = hf.get('calibration-cdl-feb2019-Ag-Ag-6kV/rmsTransverse')
g12 = hf.get('calibration-cdl-feb2019-Ag-Ag-6kV/rotationAngle')
g13 = hf.get('calibration-cdl-feb2019-Ag-Ag-6kV/eventNumber')

g14 = hf.get('calibration-cdl-feb2019-Ag-Ag-6kV/x')
g15 = hf.get('calibration-cdl-feb2019-Ag-Ag-6kV/y')
g16 = hf.get('calibration-cdl-feb2019-Ag-Ag-6kV/charge')

g17 = hf.get('calibration-cdl-feb2019-Ag-Ag-6kV/centerX')
g18 = hf.get('calibration-cdl-feb2019-Ag-Ag-6kV/centerY')

al1 = hf.get('calibration-cdl-feb2019-Al-Al-4kV/eccentricity')
al2 = hf.get('calibration-cdl-feb2019-Al-Al-4kV/energyFromCharge')
al3 = hf.get('calibration-cdl-feb2019-Al-Al-4kV/kurtosisLongitudinal')
al4 = hf.get('calibration-cdl-feb2019-Al-Al-4kV/kurtosisTransverse')
al5 = hf.get('calibration-cdl-feb2019-Al-Al-4kV/length')
al6 = hf.get('calibration-cdl-feb2019-Al-Al-4kV/skewnessLongitudinal')
al7 = hf.get('calibration-cdl-feb2019-Al-Al-4kV/skewnessTransverse')
al8 = hf.get('calibration-cdl-feb2019-Al-Al-4kV/fractionInTransverseRms')
al9 = hf.get('calibration-cdl-feb2019-Al-Al-4kV/hits')
al10 = hf.get('calibration-cdl-feb2019-Al-Al-4kV/rmsLongitudinal')
al11 = hf.get('calibration-cdl-feb2019-Al-Al-4kV/rmsTransverse')
al12 = hf.get('calibration-cdl-feb2019-Al-Al-4kV/rotationAngle')
al_eccen = np.array(al1)
al_eFC = np.array(al2)
al_kL = np.array(al3)
al_kT = np.array(al4)
al_len = np.array(al5)
al_sL = np.array(al6)
al_sT = np.array(al7)
al_frac = np.array(al8)
al_hits = np.array(al9)
al_rmsL = np.array(al10)
al_rmsT = np.array(al11)
al_rot = np.array(al12)

c1 = hf.get('calibration-cdl-feb2019-C-EPIC-0.6kV/eccentricity')
c2 = hf.get('calibration-cdl-feb2019-C-EPIC-0.6kV/energyFromCharge')
c3 = hf.get('calibration-cdl-feb2019-C-EPIC-0.6kV/kurtosisLongitudinal')
c4 = hf.get('calibration-cdl-feb2019-C-EPIC-0.6kV/kurtosisTransverse')
c5 = hf.get('calibration-cdl-feb2019-C-EPIC-0.6kV/length')
c6 = hf.get('calibration-cdl-feb2019-C-EPIC-0.6kV/skewnessLongitudinal')
c7 = hf.get('calibration-cdl-feb2019-C-EPIC-0.6kV/skewnessTransverse')
c8 = hf.get('calibration-cdl-feb2019-C-EPIC-0.6kV/fractionInTransverseRms')
c9 = hf.get('calibration-cdl-feb2019-C-EPIC-0.6kV/hits')
c10 = hf.get('calibration-cdl-feb2019-C-EPIC-0.6kV/rmsLongitudinal')
c11 = hf.get('calibration-cdl-feb2019-C-EPIC-0.6kV/rmsTransverse')
c12 = hf.get('calibration-cdl-feb2019-C-EPIC-0.6kV/rotationAngle')
c_eccen = np.array(c1)
c_eFC = np.array(c2)
c_kL = np.array(c3)
c_kT = np.array(c4)
c_len = np.array(c5)
c_sL = np.array(c6)
c_sT = np.array(c7)
c_frac = np.array(c8)
c_hits = np.array(c9)
c_rmsL = np.array(c10)
c_rmsT = np.array(c11)
c_rot = np.array(c12)

cu1_1 = hf.get('calibration-cdl-feb2019-Cu-EPIC-0.9kV/eccentricity')
cu1_2 = hf.get('calibration-cdl-feb2019-Cu-EPIC-0.9kV/energyFromCharge')
cu1_3 = hf.get('calibration-cdl-feb2019-Cu-EPIC-0.9kV/kurtosisLongitudinal')
cu1_4 = hf.get('calibration-cdl-feb2019-Cu-EPIC-0.9kV/kurtosisTransverse')
cu1_5 = hf.get('calibration-cdl-feb2019-Cu-EPIC-0.9kV/length')
cu1_6 = hf.get('calibration-cdl-feb2019-Cu-EPIC-0.9kV/skewnessLongitudinal')
cu1_7 = hf.get('calibration-cdl-feb2019-Cu-EPIC-0.9kV/skewnessTransverse')
cu1_8 = hf.get('calibration-cdl-feb2019-Cu-EPIC-0.9kV/fractionInTransverseRms')
cu1_9 = hf.get('calibration-cdl-feb2019-Cu-EPIC-0.9kV/hits')
cu1_10 = hf.get('calibration-cdl-feb2019-Cu-EPIC-0.9kV/rmsLongitudinal')
cu1_11 = hf.get('calibration-cdl-feb2019-Cu-EPIC-0.9kV/rmsTransverse')
cu1_12 = hf.get('calibration-cdl-feb2019-Cu-EPIC-0.9kV/rotationAngle')
cu1_13 = hf.get('calibration-cdl-feb2019-Cu-EPIC-0.9kV/likelihood') # likelihood method
cu1_eccen = np.array(cu1_1)
cu1_eFC = np.array(cu1_2)
cu1_kL = np.array(cu1_3)
cu1_kT = np.array(cu1_4)
cu1_len = np.array(cu1_5)
cu1_sL = np.array(cu1_6)
cu1_sT = np.array(cu1_7)
cu1_frac = np.array(cu1_8)
cu1_hits = np.array(cu1_9)
cu1_rmsL = np.array(cu1_10)
cu1_rmsT = np.array(cu1_11)
cu1_rot = np.array(cu1_12)

cu1_likeli = np.array(cu1_13) # likelihood method

cu2_1 = hf.get('calibration-cdl-feb2019-Cu-EPIC-2kV/eccentricity')
cu2_2 = hf.get('calibration-cdl-feb2019-Cu-EPIC-2kV/energyFromCharge')
cu2_3 = hf.get('calibration-cdl-feb2019-Cu-EPIC-2kV/kurtosisLongitudinal')
cu2_4 = hf.get('calibration-cdl-feb2019-Cu-EPIC-2kV/kurtosisTransverse')
cu2_5 = hf.get('calibration-cdl-feb2019-Cu-EPIC-2kV/length')
cu2_6 = hf.get('calibration-cdl-feb2019-Cu-EPIC-2kV/skewnessLongitudinal')
cu2_7 = hf.get('calibration-cdl-feb2019-Cu-EPIC-2kV/skewnessTransverse')
cu2_8 = hf.get('calibration-cdl-feb2019-Cu-EPIC-2kV/fractionInTransverseRms')
cu2_9 = hf.get('calibration-cdl-feb2019-Cu-EPIC-2kV/hits')
cu2_10 = hf.get('calibration-cdl-feb2019-Cu-EPIC-2kV/rmsLongitudinal')
cu2_11 = hf.get('calibration-cdl-feb2019-Cu-EPIC-2kV/rmsTransverse')
cu2_12 = hf.get('calibration-cdl-feb2019-Cu-EPIC-2kV/rotationAngle')
cu2_eccen = np.array(cu2_1)
cu2_eFC = np.array(cu2_2)
cu2_kL = np.array(cu2_3)
cu2_kT = np.array(cu2_4)
cu2_len = np.array(cu2_5)
cu2_sL = np.array(cu2_6)
cu2_sT = np.array(cu2_7)
cu2_frac = np.array(cu2_8)
cu2_hits = np.array(cu2_9)
cu2_rmsL = np.array(cu2_10)
cu2_rmsT = np.array(cu2_11)
cu2_rot = np.array(cu2_12)

cn1 = hf.get('calibration-cdl-feb2019-Cu-Ni-15kV/eccentricity')
cn2 = hf.get('calibration-cdl-feb2019-Cu-Ni-15kV/energyFromCharge')
cn3 = hf.get('calibration-cdl-feb2019-Cu-Ni-15kV/kurtosisLongitudinal')
cn4 = hf.get('calibration-cdl-feb2019-Cu-Ni-15kV/kurtosisTransverse')
cn5 = hf.get('calibration-cdl-feb2019-Cu-Ni-15kV/length')
cn6 = hf.get('calibration-cdl-feb2019-Cu-Ni-15kV/skewnessLongitudinal')
cn7 = hf.get('calibration-cdl-feb2019-Cu-Ni-15kV/skewnessTransverse')
cn8 = hf.get('calibration-cdl-feb2019-Cu-Ni-15kV/fractionInTransverseRms')
cn9 = hf.get('calibration-cdl-feb2019-Cu-Ni-15kV/hits')
cn10 = hf.get('calibration-cdl-feb2019-Cu-Ni-15kV/rmsLongitudinal')
cn11 = hf.get('calibration-cdl-feb2019-Cu-Ni-15kV/rmsTransverse')
cn12 = hf.get('calibration-cdl-feb2019-Cu-Ni-15kV/rotationAngle')
cn_eccen = np.array(cn1)
cn_eFC = np.array(cn2)
cn_kL = np.array(cn3)
cn_kT = np.array(cn4)
cn_len = np.array(cn5)
cn_sL = np.array(cn6)
cn_sT = np.array(cn7)
cn_frac = np.array(cn8)
cn_hits = np.array(cn9)
cn_rmsL = np.array(cn10)
cn_rmsT = np.array(cn11)
cn_rot = np.array(cn12)


mc1 = hf.get('calibration-cdl-feb2019-Mn-Cr-12kV/eccentricity')
mc2 = hf.get('calibration-cdl-feb2019-Mn-Cr-12kV/energyFromCharge')
mc3 = hf.get('calibration-cdl-feb2019-Mn-Cr-12kV/kurtosisLongitudinal')
mc4 = hf.get('calibration-cdl-feb2019-Mn-Cr-12kV/kurtosisTransverse')
mc5 = hf.get('calibration-cdl-feb2019-Mn-Cr-12kV/length')
mc6 = hf.get('calibration-cdl-feb2019-Mn-Cr-12kV/skewnessLongitudinal')
mc7 = hf.get('calibration-cdl-feb2019-Mn-Cr-12kV/skewnessTransverse')
mc8 = hf.get('calibration-cdl-feb2019-Mn-Cr-12kV/fractionInTransverseRms')
mc9 = hf.get('calibration-cdl-feb2019-Mn-Cr-12kV/hits')
mc10 = hf.get('calibration-cdl-feb2019-Mn-Cr-12kV/rmsLongitudinal')
mc11 = hf.get('calibration-cdl-feb2019-Mn-Cr-12kV/rmsTransverse')
mc12 = hf.get('calibration-cdl-feb2019-Mn-Cr-12kV/rotationAngle')
mc_eccen = np.array(mc1)
mc_eFC = np.array(mc2)
mc_kL = np.array(mc3)
mc_kT = np.array(mc4)
mc_len = np.array(mc5)
mc_sL = np.array(mc6)
mc_sT = np.array(mc7)
mc_frac = np.array(mc8)
mc_hits = np.array(mc9)
mc_rmsL = np.array(mc10)
mc_rmsT = np.array(mc11)
mc_rot = np.array(mc12)

ti1 = hf.get('calibration-cdl-feb2019-Ti-Ti-9kV/eccentricity')
ti2 = hf.get('calibration-cdl-feb2019-Ti-Ti-9kV/energyFromCharge')
ti3 = hf.get('calibration-cdl-feb2019-Ti-Ti-9kV/kurtosisLongitudinal')
ti4 = hf.get('calibration-cdl-feb2019-Ti-Ti-9kV/kurtosisTransverse')
ti5 = hf.get('calibration-cdl-feb2019-Ti-Ti-9kV/length')
ti6 = hf.get('calibration-cdl-feb2019-Ti-Ti-9kV/skewnessLongitudinal')
ti7 = hf.get('calibration-cdl-feb2019-Ti-Ti-9kV/skewnessTransverse')
ti8 = hf.get('calibration-cdl-feb2019-Ti-Ti-9kV/fractionInTransverseRms')
ti9 = hf.get('calibration-cdl-feb2019-Ti-Ti-9kV/hits')
ti10 = hf.get('calibration-cdl-feb2019-Ti-Ti-9kV/rmsLongitudinal')
ti11 = hf.get('calibration-cdl-feb2019-Ti-Ti-9kV/rmsTransverse')
ti12 = hf.get('calibration-cdl-feb2019-Ti-Ti-9kV/rotationAngle')
ti_eccen = np.array(ti1)
ti_eFC = np.array(ti2)
ti_kL = np.array(ti3)
ti_kT = np.array(ti4)
ti_len = np.array(ti5)
ti_sL = np.array(ti6)
ti_sT = np.array(ti7)
ti_frac = np.array(ti8)
ti_hits = np.array(ti9)
ti_rmsL = np.array(ti10)
ti_rmsT = np.array(ti11)
ti_rot = np.array(ti12)

#print(g13.shape)
#print(g16.shape)

g_eccen = np.array(g1)
g_eFC = np.array(g2)
g_kL = np.array(g3)
g_kT = np.array(g4)
g_len = np.array(g5)
g_sL = np.array(g6)
g_sT = np.array(g7)
g_frac = np.array(g8)
g_hits = np.array(g9)
g_rmsL = np.array(g10)
g_rmsT = np.array(g11)
g_rot = np.array(g12)
g_eventN = np.array(g13)
g_x = np.array(g14)
g_y = np.array(g15)
g_charge = np.array(g16)
g_centerX = np.array(g17) 
g_centerY = np.array(g18)

hf.close


hf_rec = h5py.File('../Dongjin/Cast_IAXO_data/reco_186.h5')
r1 = hf_rec.get('reconstruction/run_186/chip_3/eccentricity')
r2 = hf_rec.get('reconstruction/run_186/chip_3/energyFromCharge')
r3 = hf_rec.get('reconstruction/run_186/chip_3/kurtosisLongitudinal')
r4 = hf_rec.get('reconstruction/run_186/chip_3/kurtosisTransverse')
r5 = hf_rec.get('reconstruction/run_186/chip_3/length')
r6 = hf_rec.get('reconstruction/run_186/chip_3/skewnessLongitudinal')
r7 = hf_rec.get('reconstruction/run_186/chip_3/skewnessTransverse')
r8 = hf_rec.get('reconstruction/run_186/chip_3/fractionInTransverseRms')
r9 = hf_rec.get('reconstruction/run_186/chip_3/hits')
r10 = hf_rec.get('reconstruction/run_186/chip_3/rmsLongitudinal')
r11 = hf_rec.get('reconstruction/run_186/chip_3/rmsTransverse')
r12 = hf_rec.get('reconstruction/run_186/chip_3/rotationAngle')
r13 = hf_rec.get('reconstruction/run_186/chip_3/eventNumber')
r14 = hf_rec.get('reconstruction/run_186/chip_3/centerX')
r15 = hf_rec.get('reconstruction/run_186/chip_3/centerY')
r16 = hf_rec.get('reconstruction/run_186/chip_3/x')
r17 = hf_rec.get('reconstruction/run_186/chip_3/y')
r18 = hf_rec.get('reconstruction/run_186/chip_3/charge')

r_eccen = np.array(r1)
r_eFC = np.array(r2)
r_kL = np.array(r3)
r_kT = np.array(r4)
r_len = np.array(r5)
r_sL = np.array(r6)
r_sT = np.array(r7)
r_frac = np.array(r8)
r_hits = np.array(r9)
r_rmsL = np.array(r10)
r_rmsT = np.array(r11)
r_rot = np.array(r12)
r_eventN = np.array(r13)
r_centerX = np.array(r14)
r_centerY = np.array(r15)
r_x = np.array(r16)
r_y = np.array(r17)
r_charge = np.array(r18)

hf_rec.close


####

x =r_x[18090][0] 
y =r_y[18090][0]
C = r_charge[18090][0]

fig = plt.figure(figsize=(8,6))
plt.xlabel('x[pixel]')
plt.ylabel('y[pixel]')
plt.xlim([0,256])
plt.ylim([0,256])
plt.scatter(x,y, c=C, cmap='inferno',s=2)
plt.colorbar(label='Charge[electrons]')
plt.savefig('back_track_event18090')
plt.show()


xg =g_x[50][0] 
yg =g_y[50][0]
Cg = g_charge[50][0]

fig = plt.figure(figsize=(8,6))
plt.xlabel('x[pixel]')
plt.ylabel('y[pixel]')
plt.xlim([0,256])
plt.ylim([0,256])
plt.scatter(xg,yg, c=Cg, cmap='inferno',s=2)
plt.colorbar(label='Charge[electrons]')
plt.savefig('xRay_ag_event50')
plt.show()



####

y_r_eccen = np.reshape(r_eccen, (117774,))
x_r = np.arange(117774)

y_r_eventN = np.reshape(r_eventN, (117774,))
y_r_eFC = np.reshape(r_eFC, (117774,))
y_r_kL = np.reshape(r_kL, (117774,))

y_values = np.reshape(g_eccen, (178124,))
x_values = np.arange(178124)


list_of_array0 = [g_eccen, g_eFC, g_kL, g_kT, g_len, g_sL, g_sT, g_frac, g_hits, g_rmsL, g_rmsT, g_rot
                   #g_centerX, g_centerY
                   ]
list_of_array1 = [al_eccen, al_eFC, al_kL, al_kT, al_len, al_sL, al_sT, al_frac, al_hits, al_rmsL, al_rmsT, al_rot]
list_of_array2 = [c_eccen, c_eFC, c_kL, c_kT, c_len, c_sL, c_sT, c_frac, c_hits, c_rmsL, c_rmsT, c_rot]
list_of_array3 = [cu1_eccen, cu1_eFC, cu1_kL, cu1_kT, cu1_len, cu1_sL, cu1_sT, cu1_frac, cu1_hits, cu1_rmsL, cu1_rmsT, cu1_rot]
list_of_array4 = [cu2_eccen, cu2_eFC, cu2_kL, cu2_kT, cu2_len, cu2_sL, cu2_sT, cu2_frac, cu2_hits, cu2_rmsL, cu2_rmsT, cu2_rot]
list_of_array5 = [cn_eccen, cn_eFC, cn_kL, cn_kT, cn_len, cn_sL, cn_sT, cn_frac, cn_hits, cn_rmsL, cn_rmsT, cn_rot]
list_of_array6 = [mc_eccen, mc_eFC, mc_kL, mc_kT, mc_len, mc_sL, mc_sT, mc_frac, mc_hits, mc_rmsL, mc_rmsT, mc_rot]
list_of_array7 = [ti_eccen, ti_eFC, ti_kL, ti_kT, ti_len, ti_sL, ti_sT, ti_frac, ti_hits, ti_rmsL, ti_rmsT, ti_rot]

list_of_array_r = [r_eccen, r_eFC, r_kL, r_kT, r_len, r_sL, r_sT, r_frac, r_hits, r_rmsL, r_rmsT, r_rot]
list_of_array_name = ['eccentricity', 'energyFromCharge','kurtosisLongitudinal', 'kurtosisTransverse', 'length', 
                          'skewnessLongitudinal', 'skewnessTransverse', 'fractionInTransverseRms', 
                                                 'hits', 'rmsLongitudinal', 'rmsTransverse', 'rotationAngle']

def func(x):
    if x == 'hits':
        return 0, 500    
    elif x == 'eccentricity':
        return 1, 10
    elif x == 'energyFromCharge':
        return 0, 15
    elif x == 'kurtosisLongitudinal':
        return -3, 5
    elif x == 'kurtosisTransverse':
        return -2, 4
    elif x == 'length':
        return 0, 17
    elif x == 'skewnessLongitudinal':
        return -2, 2
    elif x == 'skewnessTransverse':
        return -2, 2
    elif x == 'fractionInTransverseRms':
        return 0, 0.5
    elif x == 'rmsLongitudinal':
        return 0, 5
    elif x == 'rmsTransverse':
        return 0, 2
    elif x == 'rotationAngle':
        return -0.1, 3.5
    else: 
        return None    


for i,j,n in zip(list_of_array0, list_of_array_r, list_of_array_name):
     cutrange = func(n) 
     y_r = np.reshape(j, (122138,))
     y_r = y_r[np.where(np.isfinite(y_r))[0]]
     y_r = y_r[np.where((y_r < cutrange[1]) & (y_r > cutrange[0]))[0]]

     y_g = np.reshape(i, (178124,))
     y_g = y_g[np.where(np.isfinite(y_g))[0]]
     y_g = y_g[np.where((y_g < cutrange[1]) & (y_g > cutrange[0]))[0]]

     fig, ax = plt.subplots(1, figsize = (10,10))
     #fig = plt.figure()
     plt.xlabel(n)
     plt.ylabel('event number #')
     plt.hist(y_r,100 ,alpha= 0.5, color='skyblue', lw=2, label='background')
     plt.hist(y_g,100, alpha = 0.5, color='orange', lw=2, label='photon signal')
     plt.legend()
     plt.savefig('new_cast_'+ n)
     #plt.show()


y_g = np.reshape(g_eFC, (178124,))
y_al = np.reshape(al_eFC, (131396,))
y_c = np.reshape(c_eFC, (150056,))
y_cu1 = np.reshape(cu1_eFC, (146388,))
y_cu2 = np.reshape(cu2_eFC, (123436,))
y_cn = np.reshape(cn_eFC, (46500,))
y_mc = np.reshape(mc_eFC, (48820,))
y_ti = np.reshape(ti_eFC, (166568,))

h_cut = 10
l_cut = -3
y_g = y_g[np.where((y_g < h_cut) & (y_g > l_cut))[0]]
y_al = y_al[np.where((y_al < h_cut) & (y_al > l_cut))[0]]
y_c = y_c[np.where((y_c < h_cut) & (y_c > l_cut))[0]]
y_cu1 = y_cu1[np.where((y_cu1 < h_cut) & (y_cu1 > l_cut))[0]]
y_cu2 = y_cu2[np.where((y_cu2 < h_cut) & (y_cu2 > l_cut))[0]]
y_cn = y_cn[np.where((y_cn < h_cut) & (y_cn > l_cut))[0]]
y_mc = y_mc[np.where((y_mc < h_cut) & (y_mc > l_cut))[0]]
y_ti = y_ti[np.where((y_ti < h_cut) & (y_ti > l_cut))[0]]

fig, ax = plt.subplots(1, figsize = (10,10))
plt.xlabel('')
plt.ylabel('event number')
plt.hist(y_g,100,alpha = 0.5, lw=2, label='Ag_6')
plt.hist(y_al,100,alpha = 0.5, lw=2, label='Al_4')
plt.hist(y_c,100,alpha = 0.5, lw=2, label='C_0.6')
plt.hist(y_cu1,100,alpha = 0.5, lw=2, label='Cu_0.9')
plt.hist(y_cu2,100,alpha = 0.5, lw=2, label='Cu_2')
plt.hist(y_cn,100,alpha = 0.5, lw=2, label='Cu-Ni_15')
plt.hist(y_mc,100,alpha = 0.5, lw=2, label='Mn-Cr_12')
plt.hist(y_ti,100,alpha = 0.5, lw=2, label='Ti_9')
plt.legend()
plt.savefig('energyFromCharge_for_different_energy')
#plt.savefig('eFC')
plt.show()



