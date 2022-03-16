from dataclasses import dataclass
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn
from IPython import display
import numpy as np
import math
import h5py
import pandas as pd
import matplotlib.pyplot as plt


#load the calibration data

hf = h5py.File('../Dongjin/Cast_IAXO_data/calibration-cdl-2018.h5')

# load all geom. characters together into an single array for each event  
def readDatasets(h5f, path):
    datasets = ['eccentricity',
                'energyFromCharge',
                'kurtosisLongitudinal',
                'kurtosisTransverse',
                'length',
                'skewnessLongitudinal',
                'skewnessTransverse',
                'fractionInTransverseRms',
                #'hits',
                'rmsLongitudinal',
                'rmsTransverse',
                'rotationAngle',
                'likelihood'
                ]
    data_list = [np.array(h5f.get(path + dset)).flatten() for dset in datasets]
    return np.array(data_list).transpose()

calib_dsets_Ag_n = readDatasets(hf, 'calibration-cdl-feb2019-Ag-Ag-6kV/')


# calibration datasets with different energy
calib_dsets_Cu2_n = readDatasets(hf, 'calibration-cdl-feb2019-Cu-EPIC-2kV/')
calib_dsets_Cu09_n = readDatasets(hf, 'calibration-cdl-feb2019-Cu-EPIC-0.9kV/')
calib_dsets_Al_n = readDatasets(hf, 'calibration-cdl-feb2019-Al-Al-4kV/')
calib_dsets_C_n = readDatasets(hf, 'calibration-cdl-feb2019-C-EPIC-0.6kV/')
calib_dsets_Cu_Ni_n = readDatasets(hf, 'calibration-cdl-feb2019-Cu-Ni-15kV/')
calib_dsets_Mn_Cr_n = readDatasets(hf, 'calibration-cdl-feb2019-Mn-Cr-12kV/')
calib_dsets_Ti_n = readDatasets(hf, 'calibration-cdl-feb2019-Ti-Ti-9kV/')


#load the undergrounddata reco_186 chip
hf_rec = h5py.File('../Dongjin/Cast_IAXO_data/reco_186_fixed.h5')
background_dsets = readDatasets(hf_rec, 'reconstruction/run_186/chip_3/')

# load likelihood values without cut
likeli_ag_nocut = np.array(hf.get('calibration-cdl-feb2019-Ag-Ag-6kV/likelihood'))
likeli_al_nocut = np.array(hf.get('calibration-cdl-feb2019-Al-Al-4kV/likelihood'))
likeli_c_nocut = np.array(hf.get('calibration-cdl-feb2019-C-EPIC-0.6kV/likelihood'))
likeli_cu1_nocut = np.array(hf.get('calibration-cdl-feb2019-Cu-EPIC-0.9kV/likelihood'))
likeli_cu2_nocut = np.array(hf.get('calibration-cdl-feb2019-Cu-EPIC-2kV/likelihood'))
likeli_cn_nocut = np.array(hf.get('calibration-cdl-feb2019-Cu-Ni-15kV/likelihood'))
likeli_mc_nocut = np.array(hf.get('calibration-cdl-feb2019-Mn-Cr-12kV/likelihood'))
likeli_ti_nocut = np.array(hf.get('calibration-cdl-feb2019-Ti-Ti-9kV/likelihood'))

likeli_back = np.array(hf_rec.get('reconstruction/run_186/chip_3/likelihood'))

hf.close()
hf_rec.close()


# calibration data cut

calib_dsets_Ag = calib_dsets_Ag_n[:40000]
calib_dsets_Cu2 = calib_dsets_Cu2_n[:40000]
calib_dsets_Al = calib_dsets_Al_n[:40000]
calib_dsets_C = calib_dsets_C_n[:40000]
calib_dsets_Cu09 = calib_dsets_Cu09_n[:40000]
calib_dsets_Cu_Ni = calib_dsets_Cu_Ni_n[:40000]
calib_dsets_Mn_Cr = calib_dsets_Mn_Cr_n[:40000]
calib_dsets_Ti = calib_dsets_Ti_n[:40000]


calib_dsets = np.concatenate((calib_dsets_Ag, calib_dsets_Al, calib_dsets_C, 
                            calib_dsets_Cu09, 
                            calib_dsets_Cu2, calib_dsets_Cu_Ni, calib_dsets_Mn_Cr, calib_dsets_Ti
                            ),axis=0)

# try to cut the datasets with some cutvalues
df = pd.DataFrame(calib_dsets,
                   columns=['eccen',
                   'eFC',
                   'kL','kT','len','sL','sT','frac',#'hits',
                   'rmsL','rmsT'
                   ,'rot',
                   'likelihood'
                   ])
# the cut values are defined
dfc_cut = df[ ((df['eccen']>1) & (df['eccen']<5)) & ((df['eFC']>0) & (df['eFC']<15)) &
              ((df['kL']>-2) & (df['kL']<5)) & ((df['kT']>-2) & (df['kT']<4)) & ((df['len']>0) & (df['len']<14)) &
              ((df['sL']>-2) & (df['sL']<2)) & ((df['sT']>-2) & (df['sT']<2)) & ((df['frac']>0) & (df['frac']<0.5)) &
              #((df['hits']>0) & (df['hits']<500)) & 
              ((df['rmsL']>0) & (df['rmsL']<4)) &
              ((df['rmsT']>0) & (df['rmsT']<2)) &
                ((df['rot']>-0.1) & (df['rot']<3.5))
            ]
dfc_cut = dfc_cut.drop('eFC', axis=1)
dfc_cut = dfc_cut.drop('likelihood', axis=1)


# background data cut
dfb = pd.DataFrame(background_dsets,
                   columns=['eccen','eFC',
                   'kL','kT','len','sL','sT','frac',#'hits',
                   'rmsL','rmsT'
                   ,'rot',
                   'likelihood'
                   ])
dfb_cut = dfb[ ((dfb['eccen']>0.0) & (dfb['eccen']<10)) & ((dfb['eFC']>0.0) & (dfb['eFC']<5)) &
               ((dfb['kL']>-2) & (dfb['kL']<4)) & ((dfb['kT']>-2) & (dfb['kT']<4)) &
               ((dfb['len']>0) & (dfb['len']<18)) & ((dfb['sL']>-2) & (dfb['sL']<2)) &
               ((dfb['sT']>-2) & (dfb['sT']<2)) & ((dfb['frac']>-1) & (dfb['frac']<1)) &
               #((dfb['hits']>0) & (dfb['hits']<500)) & 
               ((dfb['rmsL']>0) & (dfb['rmsL']<5)) &
               ((dfb['rmsT']>0) & (dfb['rmsT']<2)) ]

dfb_cut = dfb_cut.drop('eFC', axis=1)
dfb_cut = dfb_cut.drop('likelihood', axis=1)

# validation data cut
dfv = pd.DataFrame(calib_dsets,
                    columns=['eccen','eFC',
                    'kL','kT','len','sL','sT','frac',#'hits',
                    'rmsL','rmsT'
                    ,'rot', 'likelihood'
                    ])

dfv_E = dfv.drop('likelihood', axis=1)

#dfv = dfv.drop('eFC', axis=1)

dfv_cut = dfv[ ((dfv['eccen']>0.0) & (dfv['eccen']<5)) & #((dfv['eFC']>0.0) & (dfv['eFC']<15)) &
               ((dfv['kL']>-2) & (dfv['kL']<5)) & ((dfv['kT']>-2) & (dfv['kT']<4)) &
               ((dfv['len']>0) & (dfv['len']<18)) & ((dfv['sL']>-2) & (dfv['sL']<2)) &
               ((dfv['sT']>-2) & (dfv['sT']<2)) & ((dfv['frac']>0) & (dfv['frac']<1)) &
               #((dfv['hits']>0) & (dfv['hits']<500)) & 
               ((dfv['rmsL']>0) & (dfv['rmsL']<5)) &
               ((dfv['rmsT']>0) & (dfv['rmsT']<2)) ]

'''
# get LogL data which passed through cut 
df_LogL = dfv_cut['likelihood']
likeli_Ag = df_LogL.to_numpy().astype(np.float32)
'''
dfv_cut = dfv_cut.drop('eFC', axis=1)
dfv_cut = dfv_cut.drop('likelihood', axis=1)


# cut for validation and LogL 
def validation_cut(calib_dsets):
    dfv = pd.DataFrame(calib_dsets,
                    columns=['eccen','eFC',
                    'kL','kT','len','sL','sT','frac',#'hits',
                    'rmsL','rmsT'
                    ,'rot', 'likelihood'
                    ])

    dfv_cut = dfv[  ((dfv['eccen']>1) & (dfv['eccen']<2.5)) & #((dfv['eFC']>0) & (dfv['eFC']<15)) &
              ((dfv['kL']>-2) & (dfv['kL']<5)) & ((dfv['kT']>-2) & (dfv['kT']<4)) & ((dfv['len']>0) & (dfv['len']<14)) &
              ((dfv['sL']>-2) & (dfv['sL']<2)) & ((dfv['sT']>-2) & (dfv['sT']<2)) & ((dfv['frac']>0) & (dfv['frac']<0.5)) &
              #((df['hits']>0) & (df['hits']<500)) & 
              ((dfv['rmsL']>0) & (dfv['rmsL']<4)) &
              ((dfv['rmsT']>0) & (dfv['rmsT']<2)) & ((dfv['rot']>-0.1) & (dfv['rot']<3.5))  ]
    
    # get LogL data which passed through cut 
    df_LogL = dfv_cut['likelihood']
    likeli_Ag = df_LogL.to_numpy().astype(np.float32)

    dfv_cut = dfv_cut.drop('eFC', axis=1)
    dfv_cut = dfv_cut.drop('likelihood', axis=1)

    return dfv_cut.to_numpy().astype(np.float32), likeli_Ag



def shuffleAndFilter(t):
    # filter out every rows which contains a NaN value in the dataset
    t = t[~np.any(np.isnan(t), axis = 1)]
    seed = 10
    np.random.seed(seed)
    np.random.shuffle(t)
    return t

# now get data from DF and shuffle & filter for NaN
calibration_data = shuffleAndFilter(dfc_cut.to_numpy().astype(np.float32))
background_data  = shuffleAndFilter(dfb_cut.to_numpy().astype(np.float32))
validation_data_all = shuffleAndFilter(dfv_cut.to_numpy().astype(np.float32)) 


 # cdl validation data cut applied
validation_data_ag = shuffleAndFilter(validation_cut(calib_dsets_Ag)[0])
validation_data_al = shuffleAndFilter(validation_cut(calib_dsets_Al)[0])
validation_data_c = shuffleAndFilter(validation_cut(calib_dsets_C)[0])
validation_data_cu1 = shuffleAndFilter(validation_cut(calib_dsets_Cu09)[0])
validation_data_cu2 = shuffleAndFilter(validation_cut(calib_dsets_Cu2)[0])
validation_data_cn = shuffleAndFilter(validation_cut(calib_dsets_Cu_Ni)[0])
validation_data_mc = shuffleAndFilter(validation_cut(calib_dsets_Mn_Cr)[0])
validation_data_ti = shuffleAndFilter(validation_cut(calib_dsets_Ti)[0])


 # cdl logL data cut applied
likeli_ag = validation_cut(calib_dsets_Ag_n)[1]
likeli_al = validation_cut(calib_dsets_Al_n)[1]
likeli_c = validation_cut(calib_dsets_C_n)[1]
likeli_cu09 = validation_cut(calib_dsets_Cu09_n)[1]
likeli_cu2 = validation_cut(calib_dsets_Cu2_n)[1]
likeli_cn = validation_cut(calib_dsets_Cu_Ni_n)[1]
likeli_mc = validation_cut(calib_dsets_Mn_Cr_n)[1]
likeli_ti = validation_cut(calib_dsets_Ti_n)[1]



#divide the data for training and test
train_data = calibration_data[:70000]
test_data = calibration_data[70001:80000]

train_data_background = background_data[:70000]
test_data_background = background_data[70001:80000]



####
# in this part only the chracter 'energyFromCharge' is used to create a dataset with same eventnumber for all energy area

# pass through only the 'energyFromCharge' values within min and max 
def energycutValidation(df, min_cut, max_cut, num_data):
    df = df[ ((dfv['eFC']>=min_cut) & (dfv['eFC']<=max_cut)) ]
    df = df.to_numpy().astype(np.float32)
    seed = 10
    np.random.seed(seed)
    np.random.shuffle(df)
    df = df[:num_data]
    return df

n_e = 2000 
 # cdl data with only cut in the selected energy area
 # devide the energy area with range of 0 to 8 keV into 32 parts with an intervall of 0.25 keV 
valid_data1 = energycutValidation(dfv_E, 0, 0.25, n_e)
valid_data2 = energycutValidation(dfv_E, 0.25, 0.5, n_e)
valid_data3 = energycutValidation(dfv_E, 0.5, 0.75, n_e)
valid_data4 = energycutValidation(dfv_E, 0.75, 1, n_e)
valid_data5 = energycutValidation(dfv_E, 1, 1.25, n_e)
valid_data6 = energycutValidation(dfv_E, 1.25, 1.5, n_e)
valid_data7 = energycutValidation(dfv_E, 1.5, 1.75, n_e)
valid_data8 = energycutValidation(dfv_E, 1.75, 2, n_e)
valid_data9 = energycutValidation(dfv_E, 2, 2.25, n_e)
valid_data10 = energycutValidation(dfv_E, 2.25, 2.5, n_e)
valid_data11 = energycutValidation(dfv_E, 2.5, 2.75, n_e)
valid_data12 = energycutValidation(dfv_E, 2.75, 3, n_e)
valid_data13 = energycutValidation(dfv_E, 3, 3.25, n_e)
valid_data14 = energycutValidation(dfv_E, 3.25, 3.5, n_e)
valid_data15 = energycutValidation(dfv_E, 3.5, 3.75, n_e)
valid_data16 = energycutValidation(dfv_E, 3.75, 4, n_e)
valid_data17 = energycutValidation(dfv_E, 4, 4.25, n_e)
valid_data18 = energycutValidation(dfv_E, 4.25, 4.5, n_e)
valid_data19 = energycutValidation(dfv_E, 4.5, 4.75, n_e)
valid_data20 = energycutValidation(dfv_E, 4.75, 5, n_e)
valid_data21 = energycutValidation(dfv_E, 5, 5.25, n_e)
valid_data22 = energycutValidation(dfv_E, 5.25, 5.5, n_e)
valid_data23 = energycutValidation(dfv_E, 5.5, 5.75, n_e)
valid_data24 = energycutValidation(dfv_E, 5.75, 6, n_e)
valid_data25 = energycutValidation(dfv_E, 6, 6.25, n_e)
valid_data26 = energycutValidation(dfv_E, 6.25, 6.5, n_e)
valid_data27 = energycutValidation(dfv_E, 6.5, 6.75, n_e)
valid_data28 = energycutValidation(dfv_E, 6.75, 7, n_e)
valid_data29 = energycutValidation(dfv_E, 7, 7.25, n_e)
valid_data30 = energycutValidation(dfv_E, 7.25, 7.5, n_e)
valid_data31 = energycutValidation(dfv_E, 7.5, 7.75, n_e)
valid_data32 = energycutValidation(dfv_E, 7.75, 8, n_e)

dset = np.concatenate((#valid_data1, 
valid_data2, valid_data3, valid_data4, valid_data5, valid_data6,
                       valid_data7, valid_data8, valid_data9, valid_data10, valid_data11, valid_data12 
                       ,valid_data13, valid_data14, valid_data15, valid_data16, valid_data17, valid_data18
                       ,valid_data19, valid_data20, valid_data21, valid_data22 
                       ,valid_data23, valid_data24, valid_data25, valid_data26, valid_data27, valid_data28
                       ,valid_data29, valid_data30, valid_data31, valid_data32
                       ))

#plot the flat energydistribution of inputdata
dset = dset[:, 1] #taking only energyFromCharge values 
plt.xlabel('E')
plt.ylabel('event number')
plt.hist(dset,100,alpha = 0.5, lw=2, label='')
plt.show()

# cut off the 'energyFromCharge' for the validation of the network 
valid_data1= np.delete(valid_data1,1,1)
valid_data2= np.delete(valid_data2,1,1)
valid_data3= np.delete(valid_data3,1,1)
valid_data4= np.delete(valid_data4,1,1)
valid_data5= np.delete(valid_data5,1,1)
valid_data6= np.delete(valid_data6,1,1)
valid_data7= np.delete(valid_data7,1,1)
valid_data8= np.delete(valid_data8,1,1)
valid_data9= np.delete(valid_data9,1,1)
valid_data10= np.delete(valid_data10,1,1)
valid_data11= np.delete(valid_data11,1,1)
valid_data12= np.delete(valid_data12,1,1)
valid_data13= np.delete(valid_data13,1,1)
valid_data14= np.delete(valid_data14,1,1)
valid_data15= np.delete(valid_data15,1,1)
valid_data16= np.delete(valid_data16,1,1)
valid_data17= np.delete(valid_data17,1,1)
valid_data18= np.delete(valid_data18,1,1)
valid_data19= np.delete(valid_data19,1,1)
valid_data20= np.delete(valid_data20,1,1)
valid_data21= np.delete(valid_data21,1,1)
valid_data22= np.delete(valid_data22,1,1)
valid_data23= np.delete(valid_data23,1,1)
valid_data24= np.delete(valid_data24,1,1)
valid_data25= np.delete(valid_data25,1,1)
valid_data26= np.delete(valid_data26,1,1)
valid_data27= np.delete(valid_data27,1,1)
valid_data28= np.delete(valid_data28,1,1)
valid_data29= np.delete(valid_data29,1,1)
valid_data30= np.delete(valid_data30,1,1)
valid_data31= np.delete(valid_data31,1,1)
valid_data32= np.delete(valid_data32,1,1)


#valid_data = calibration_data[50001:51000]
valid_data = calibration_data[80001:90001]
valid_data_back = background_data[80001:90001]


# prepare dataset with labels
class Dataset(data.Dataset):
    'Characterizes a dataset for Pytorch'
    def __init__(self, train_data, labels):
        'Initialization'
        self.labels = labels
        self.train_data = torch.tensor(train_data)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.train_data)

    def __getitem__(self, index):
        'Generates one sample of data'
        #select sample
        ID = self.train_data[index]

        # Load data and get label
        X = ID
        if self.labels == 0:
            y = np.array([1.0,0.0])
        elif self.labels == 1:
            y = np.array([0.0,1.0])
        # y = self.labels
        return X, y

label_calibration = 1
label_background = 0

#select the batch size
batch_size = 100


####
# this part reshape the datasets into a shape which the network demands

training_set1 = Dataset(train_data, label_calibration)
training_set2 = Dataset(train_data_background, label_background)
training_set = data.ConcatDataset([training_set2, training_set1])

train_iter = data.DataLoader(training_set, batch_size, shuffle=True,
                             num_workers = 0)

test_set1 = Dataset(test_data, label_calibration)
test_set2 = Dataset(test_data_background, label_background)
test_set = data.ConcatDataset([test_set1, test_set2])

test_iter = data.DataLoader(test_set, batch_size, shuffle=True, num_workers = 0)

def setup_validation_calib(d, x):
    f = Dataset(d[:x], label_calibration)
    m = data.DataLoader(f, batch_size, shuffle=True, num_workers = 0)
    return m
 
num_vset = 10000

valid_iter_ag =setup_validation_calib(validation_data_ag, num_vset)
valid_iter_al = setup_validation_calib(validation_data_al, num_vset)
valid_iter_c = setup_validation_calib(validation_data_c, num_vset)
valid_iter_cu1 = setup_validation_calib(validation_data_cu1, num_vset)
valid_iter_cu2 = setup_validation_calib(validation_data_cu2, num_vset)
valid_iter_cn = setup_validation_calib(validation_data_cn,num_vset)
valid_iter_mc = setup_validation_calib(validation_data_mc, num_vset)
valid_iter_ti = setup_validation_calib(validation_data_ti, num_vset)

validation_set1 = Dataset(valid_data, label_calibration)
validation_set2 = Dataset(valid_data_back, label_background)
validation_set = data.ConcatDataset([validation_set1, validation_set2])
valid_iter_cal = data.DataLoader(validation_set1, batch_size, shuffle=True, num_workers = 0)
valid_iter_back = data.DataLoader(validation_set2, batch_size, shuffle=True, num_workers = 0)
valid_iter = data.DataLoader(validation_set, batch_size, shuffle=True, num_workers=0)



# validation sets with flat energy distribution 
valid_set1 = Dataset(valid_data1, label_calibration)
valid_set2 = Dataset(valid_data2, label_calibration)
valid_set3 = Dataset(valid_data3, label_calibration)
valid_set4 = Dataset(valid_data4, label_calibration)
valid_set5 = Dataset(valid_data5, label_calibration)
valid_set6 = Dataset(valid_data6, label_calibration)
valid_set7 = Dataset(valid_data7, label_calibration)
valid_set8 = Dataset(valid_data8, label_calibration)
valid_set9 = Dataset(valid_data9, label_calibration)
valid_set10 = Dataset(valid_data10, label_calibration)
valid_set11 = Dataset(valid_data11, label_calibration)
valid_set12 = Dataset(valid_data12, label_calibration)
valid_set13 = Dataset(valid_data13, label_calibration)
valid_set14 = Dataset(valid_data14, label_calibration)
valid_set15 = Dataset(valid_data15, label_calibration)
valid_set16 = Dataset(valid_data16, label_calibration)
valid_set17 = Dataset(valid_data17, label_calibration)
valid_set18 = Dataset(valid_data18, label_calibration)
valid_set19 = Dataset(valid_data19, label_calibration)
valid_set20 = Dataset(valid_data20, label_calibration)
valid_set21 = Dataset(valid_data21, label_calibration)
valid_set22 = Dataset(valid_data22, label_calibration)
valid_set23 = Dataset(valid_data23, label_calibration)
valid_set24 = Dataset(valid_data24, label_calibration)
valid_set25 = Dataset(valid_data25, label_calibration)
valid_set26 = Dataset(valid_data26, label_calibration)
valid_set27 = Dataset(valid_data27, label_calibration)
valid_set28 = Dataset(valid_data28, label_calibration)
valid_set29 = Dataset(valid_data29, label_calibration)
valid_set30 = Dataset(valid_data30, label_calibration)
valid_set31 = Dataset(valid_data31, label_calibration)
valid_set32 = Dataset(valid_data32, label_calibration)

valid_iter1 = data.DataLoader(valid_set1, batch_size, shuffle=True, num_workers = 0)
valid_iter2 = data.DataLoader(valid_set2, batch_size, shuffle=True, num_workers = 0)
valid_iter3 = data.DataLoader(valid_set3, batch_size, shuffle=True, num_workers = 0)
valid_iter4 = data.DataLoader(valid_set4, batch_size, shuffle=True, num_workers = 0)
valid_iter5 = data.DataLoader(valid_set5, batch_size, shuffle=True, num_workers = 0)
valid_iter6 = data.DataLoader(valid_set6, batch_size, shuffle=True, num_workers = 0)
valid_iter7 = data.DataLoader(valid_set7, batch_size, shuffle=True, num_workers = 0)
valid_iter8 = data.DataLoader(valid_set8, batch_size, shuffle=True, num_workers = 0)
valid_iter9 = data.DataLoader(valid_set9, batch_size, shuffle=True, num_workers = 0)
valid_iter10 = data.DataLoader(valid_set10, batch_size, shuffle=True, num_workers = 0)
valid_iter11 = data.DataLoader(valid_set11, batch_size, shuffle=True, num_workers = 0)
valid_iter12 = data.DataLoader(valid_set12, batch_size, shuffle=True, num_workers = 0)
valid_iter13 = data.DataLoader(valid_set13, batch_size, shuffle=True, num_workers = 0)
valid_iter14 = data.DataLoader(valid_set14, batch_size, shuffle=True, num_workers = 0)
valid_iter15 = data.DataLoader(valid_set15, batch_size, shuffle=True, num_workers = 0)
valid_iter16 = data.DataLoader(valid_set16, batch_size, shuffle=True, num_workers = 0)
valid_iter17 = data.DataLoader(valid_set17, batch_size, shuffle=True, num_workers = 0)
valid_iter18 = data.DataLoader(valid_set18, batch_size, shuffle=True, num_workers = 0)
valid_iter19 = data.DataLoader(valid_set19, batch_size, shuffle=True, num_workers = 0)
valid_iter20 = data.DataLoader(valid_set20, batch_size, shuffle=True, num_workers = 0)
valid_iter21 = data.DataLoader(valid_set21, batch_size, shuffle=True, num_workers = 0)
valid_iter22 = data.DataLoader(valid_set22, batch_size, shuffle=True, num_workers = 0)
valid_iter23 = data.DataLoader(valid_set23, batch_size, shuffle=True, num_workers = 0)
valid_iter24 = data.DataLoader(valid_set24, batch_size, shuffle=True, num_workers = 0)
valid_iter25 = data.DataLoader(valid_set25, batch_size, shuffle=True, num_workers = 0)
valid_iter26 = data.DataLoader(valid_set26, batch_size, shuffle=True, num_workers = 0)
valid_iter27 = data.DataLoader(valid_set27, batch_size, shuffle=True, num_workers = 0)
valid_iter28 = data.DataLoader(valid_set28, batch_size, shuffle=True, num_workers = 0)
valid_iter29 = data.DataLoader(valid_set29, batch_size, shuffle=True, num_workers = 0)
valid_iter30 = data.DataLoader(valid_set30, batch_size, shuffle=True, num_workers = 0)
valid_iter31 = data.DataLoader(valid_set31, batch_size, shuffle=True, num_workers = 0)
valid_iter32 = data.DataLoader(valid_set32, batch_size, shuffle=True, num_workers = 0)
####


####
#starting mlp

seed = 1
torch.manual_seed(seed)

# define the MLP model 
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(),
                                    nn.Linear(10, 500),
                                    nn.ReLU(),
                                    nn.Linear(500, 300),
                                    nn.ReLU(),
                                    nn.Linear(300,2)
                                    )
    def forward(self, x):
        return self.layers(x)

net = MLP()

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

# NOTE: one should check here, whether this normalization is
# actually the most efficient one etc!
net.apply(init_weights);

# define the loss function sigmoid + cross entropy loss
loss = nn.BCEWithLogitsLoss()
#loss = nn.CrossEntropyLoss()


class Accumulator:  #@save
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat, y): 
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
        y = y.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):  
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    correct = 0

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
            y_hat = net.forward(X)
            pred = y_hat.argmax(1)
            correct += pred.eq(y.argmax(1)).sum()
    #print("Test Accuracy ", correct / len(data_iter))
    return metric[0] / metric[1], correct / (len(data_iter)*batch_size)


# train the model for an epoch
def train_epoch_ch3(net, train_iter, loss, updater):  
    """The training loop"""
    # Set the model to training mode
    if isinstance(net, torch.nn.Module):
        net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    correct = 0
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net.forward(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # Using PyTorch in-built optimizer & loss criterion
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.numel())
        else:
            # Using custom built optimizer & loss criterion
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        ## Now compute the accuracy manually.
        pred = y_hat.argmax(1)
        correct += pred.eq(y.argmax(1)).sum()

    # Return training loss and training accuracy
    #if True: quit()
    #print("Train Accuracy ", correct / len(train_iter))
    return metric[0] / metric[2], metric[1] / metric[2], correct / (len(train_iter)*batch_size)


#train the model for whole epoch numbers
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """Train a model"""
    x = np.arange(num_epochs)
    y_loss = np.empty(shape=(num_epochs,))
    y_train_acc = np.empty(shape=(num_epochs,))
    y_test_acc = np.empty(shape=(num_epochs,))
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        #print('Epoch',epoch,':  ' 'train_loss=',train_metrics[0], 'train_acc=',train_metrics[2], 'test_acc=',test_acc[1])
        y_loss[epoch] = train_metrics[0]
        y_train_acc[epoch] = train_metrics[2]
        y_test_acc[epoch] = test_acc[1]
    train_loss = train_metrics[0]
    train_acc = train_metrics[2]
    #assert train_loss < 0.5, train_loss
    #assert train_acc <= 1 and train_acc > 0.5, train_acc
    #assert test_acc <= 1 and test_acc > 0.5, test_acc
    print('end result =', train_loss, train_acc, test_acc[1])
    fig, ax = plt.subplots(1, figsize = (10,10))
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    #plt.plot(x, y_loss, lw=2, label='loss')
    plt.plot(x, y_train_acc, 'r', lw=2, label='train_accuracy')
    plt.plot(x, y_test_acc, 'b', lw=2, label='test_accuracy')
    plt.legend()
    #plt.savefig('mlp_acc_70000')
    plt.show()

    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(x, y_loss, lw=2, label='loss')
    plt.legend()
    #plt.savefig('mlp_loss_70000')
    plt.show()   

    
    
# compute the accuracy of validation data after training the network
def valid_ch3(net, valid_iter):
    valid_acc = evaluate_accuracy(net, valid_iter)
    print('Validation Accuracy:', valid_acc[1])
    return valid_acc[1]


# should compute the output distributions of the mlp and the accuracy
def accuracy_validation_roc(net, data_iter, set_length):
    if isinstance(net,torch.nn.Module):
        net.eval()
    res = np.empty(shape=(set_length,2))
    batch_idx = 0
    correct = 0

    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net.forward(X)  
            pred = y_hat.argmax(1)
            correct += pred.eq(y.argmax(1)).sum()
            res[batch_idx * batch_size : (batch_idx + 1) * batch_size, :] = y_hat 
            batch_idx = batch_idx + 1

    return correct/(len(data_iter)*batch_size), res

#separate output distributions for neuron 0 and 1           
def valid_roc(net, valid_iter, set_length):
    valid_acc = accuracy_validation_roc(net, valid_iter, set_length)
    print('Validation Accuracy:', valid_acc[0])
    res = valid_acc[1]
    res0 = res[:,0]
    res1 = res[:,1]
    return res0, res1


# calculate roc-curve
def roc_curve(signal_output, background_output, m):
    back = background_output # use second neuron outputs
    signal = signal_output
    back = sorted(back)
    signal = sorted(signal)
    #print(len(back), len(signal))
    min_b = np.min(back) 
    max_b = np.max(back)
    min_s = np.min(signal)
    max_s = np.max(signal)

    if min_b < min_s:
        min = min_b
    else:
        min = min_s

    if max_b < max_s:
        max = max_s
    else:
        max = max_b

    if m == 44:
        max = 44

    num_bin = 1000
    bin_intervall = (max + abs(min)) / num_bin 
    #print(max)
    #print(min)
    roc = np.empty(shape=(num_bin,2))
    cut_idx_b = 0
    cut_idx_s = 0
    for i in range(num_bin):
        cut = min + i * bin_intervall
        for j in range(cut_idx_b ,len(back)):
            if back[j] >= cut:
                roc[i,0] = 1 - ((j-1) / len(back))
                cut_idx_b = j
                break
        for n in range(cut_idx_s ,len(signal)):
            if signal[n] >= cut:
                cut_idx_s = n
                break
        roc[i,1] = (cut_idx_s-1) / len(signal)
    #print(roc[0,:])

    # if signal is lower, roc[0] is signal efficiency
    return roc[:,0], roc[:,1] 


# select the number of epochs and the learnug rate
num_epochs, lr = 150, 0.05
updater = torch.optim.SGD(net.parameters(), lr=lr)

print('check')

# train the network
train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

 # validate the network
valid_ch3(net, valid_iter)
valid_ch3(net, valid_iter_cal)
valid_ch3(net, valid_iter_back)

# compute the output distribution for calibration and background data
output_cal = valid_roc(net, valid_iter_cal, 20000)
output_back = valid_roc(net, valid_iter_back, 20000)

 # plot the output distibutions of the output neuron 0 and 1 
plt.xlabel('output (neuron 0)')
plt.ylabel('event number #')
plt.hist(output_back[0],100,alpha = 0.5, lw=2, label='background')
plt.hist(output_cal[0],100,alpha = 0.5, lw=2, label='signal_all_calibration')
plt.legend()
#plt.savefig('validation_mlp_output0_all_cal_2')
plt.show()
plt.xlabel('output (neuron 1)')
plt.ylabel('event number #')
plt.hist(output_back[1],100,alpha = 0.5, lw=2, label= 'background')
plt.hist(output_cal[1],100,alpha = 0.5, lw=2, label= 'signal_all_calibration')
plt.legend()
#plt.savefig('validation_mlp_output1_all_cal_2')
plt.show()

 # create ROC-curve for the validation 
all = roc_curve(output_cal[0], output_back[0],13)
plt.figure()
plt.grid(axis='both', linestyle='--', linewidth=1)
plt.xlabel('signal efficiency')
plt.ylabel('background rejection')
plt.plot(all[1], all[0],c='b',lw=1,label='mlp-all energy')
plt.legend()
plt.savefig('roc_curve_all_calib_2')
plt.show()



####
# create and plot roc curves for all cdl data with LogL data
likeli_back = np.clip(likeli_back,0,50)
def plot_roc_curve_with_logL(likeli, valid_iter, name):
    likeli = likeli[np.where((likeli < 40) & (likeli > 0))[0]]
    output_cal = valid_roc(net, valid_iter, num_vset)
    output_back = valid_roc(net, valid_iter_back, num_vset)
    roc_cdl = roc_curve(output_cal[0], output_back[0], 13)
    roc_likeli = roc_curve(likeli, likeli_back, 44)
    return roc_likeli[0], roc_likeli[1], roc_cdl[0], roc_cdl[1]

l_val = [valid_iter_ag, valid_iter_al, valid_iter_c, valid_iter_cu1, valid_iter_cu2, 
        valid_iter_cn, valid_iter_mc, valid_iter_ti]

 # computing and setting for the roc-curve of the calibration and likelihood data  
ti = plot_roc_curve_with_logL(likeli_ti ,valid_iter_ti,'Ti-Ti-9kV')
ag = plot_roc_curve_with_logL(likeli_ag ,valid_iter_ag,'Ag-Ag-6kV')
al = plot_roc_curve_with_logL(likeli_al ,valid_iter_al,'Al-Al-4kV')
c = plot_roc_curve_with_logL(likeli_c,valid_iter_c,'C-EPIC-06kV')
cu1 = plot_roc_curve_with_logL(likeli_cu09 ,valid_iter_cu1,'Cu-EPIC-09kV')
cu2 = plot_roc_curve_with_logL(likeli_cu2 ,valid_iter_cu2,'Cu-EPIC-2kV')
cn = plot_roc_curve_with_logL(likeli_cn ,valid_iter_cn,'Cu-Ni-15kV')
mc = plot_roc_curve_with_logL(likeli_mc ,valid_iter_mc,'Mn-Cr-12kV')


 # plot the roc-curves of each calibration target to compare the validation and likelihood results
plt.figure()
plt.grid(axis='both', linestyle='--', linewidth=1)
plt.xlabel('signal efficiency')
plt.ylabel('background rejection')
plt.plot(ag[1], ag[0],c='b',lw=1,label='LogL')
plt.plot(ag[3], ag[2],c='b',linestyle='--',lw=1,label='mlp-Ag-6')
plt.legend()
plt.savefig('roc_curve_with_logL_ag')
plt.show()

plt.figure()
plt.grid(axis='both', linestyle='--', linewidth=1)
plt.xlabel('signal efficiency')
plt.ylabel('background rejection')
plt.plot(al[1], al[0],c='r',lw=1,label='LogL')
plt.plot(al[3], al[2],c='r',linestyle='--',lw=1,label='mlp-Al-4')
plt.legend()
plt.savefig('roc_curve_with_logL_al')

plt.figure()
plt.grid(axis='both', linestyle='--', linewidth=1)
plt.xlabel('signal efficiency')
plt.ylabel('background rejection')
plt.plot(c[1], c[0],c='g',lw=1,label='LogL')
plt.plot(c[3], c[2],c='g',linestyle='--',lw=1,label='mlp-C-0.6')
plt.legend()
plt.savefig('roc_curve_with_logL_c')

plt.figure()
plt.grid(axis='both', linestyle='--', linewidth=1)
plt.xlabel('signal efficiency')
plt.ylabel('background rejection')
plt.plot(cu1[1], cu1[0],c='m',lw=1,label='LogL')
plt.plot(cu1[3], cu1[2],c='m',linestyle='--',lw=1,label='mlp-Cu-0.9')
plt.legend()
plt.savefig('roc_curve_with_logL_cu1')

plt.figure()
plt.grid(axis='both', linestyle='--', linewidth=1)
plt.xlabel('signal efficiency')
plt.ylabel('background rejection')
plt.plot(cu2[1], cu2[0],c='c',lw=1,label='LogL')
plt.plot(cu2[3], cu2[2],c='c',linestyle='--',lw=1,label='mlp-Cu-2')
plt.legend()
plt.savefig('roc_curve_with_logL_cu2')

plt.figure()
plt.grid(axis='both', linestyle='--', linewidth=1)
plt.xlabel('signal efficiency')
plt.ylabel('background rejection')
plt.plot(cn[1], cn[0],c='y',lw=1,label='LogL')
plt.plot(cn[3], cn[2],c='y',linestyle='--',lw=1,label='mlp-Cu-Ni-15')
plt.legend()
plt.savefig('roc_curve_with_logL_cu_ni')

plt.figure()
plt.grid(axis='both', linestyle='--', linewidth=1)
plt.xlabel('signal efficiency')
plt.ylabel('background rejection')
plt.plot(mc[1], mc[0],c='tab:brown',lw=1,label='LogL')
plt.plot(mc[3], mc[2],c='tab:brown',linestyle='--',lw=1,label='mlp-Mn-Cr-12')
plt.legend()
plt.savefig('roc_curve_with_logL_mn_cr')

plt.figure()
plt.grid(axis='both', linestyle='--', linewidth=1)
plt.xlabel('signal efficiency')
plt.ylabel('background rejection')
plt.plot(ti[1], ti[0],c='k',lw=1,label='LogL')
plt.plot(ti[3], ti[2],c='k',linestyle='--',lw=1,label='mlp-Ti-9')
plt.legend()
plt.savefig('roc_curve_with_logL_ti')

 # plot the roc-curves of each calibration target together
plt.figure()
plt.grid(axis='both', linestyle='--', linewidth=1)
plt.xlabel('signal efficiency')
plt.ylabel('background rejection')
plt.plot(ag[1], ag[0],c='b',lw=1,label='LogL')
plt.plot(ag[3], ag[2],c='b',linestyle='--',lw=1,label='mlp-Ag-6')
plt.plot(al[1], al[0],c='r',lw=1,label='LogL')
plt.plot(al[3], al[2],c='r',linestyle='--',lw=1,label='mlp-Al-4')
plt.plot(c[1], c[0],c='g',lw=1,label='LogL')
plt.plot(c[3], c[2],c='g',linestyle='--',lw=1,label='mlp-C-0.6')
plt.plot(cu1[1], cu1[0],c='m',lw=1,label='LogL')
plt.plot(cu1[3], cu1[2],c='m',linestyle='--',lw=1,label='mlp-Cu-0.9')
plt.plot(cu2[1], cu2[0],c='c',lw=1,label='LogL')
plt.plot(cu2[3], cu2[2],c='c',linestyle='--',lw=1,label='mlp-Cu-2')
plt.plot(cn[1], cn[0],c='y',lw=1,label='LogL')
plt.plot(cn[3], cn[2],c='y',linestyle='--',lw=1,label='mlp-Cu-Ni-15')
plt.plot(mc[1], mc[0],c='tab:brown',lw=1,label='LogL')
plt.plot(mc[3], mc[2],c='tab:brown',linestyle='--',lw=1,label='mlp-Mn-Cr-12')
plt.plot(ti[1], ti[0],c='k',lw=1,label='LogL')
plt.plot(ti[3], ti[2],c='k',linestyle='--',lw=1,label='mlp-Ti-9')
plt.legend()
plt.savefig('roc_curves_with_logL_all_targets')
plt.show()
####

####
 # accuracy-energy distribution for the flattend energy area from 0 to 8 keV

l = [valid_iter1, valid_iter2, valid_iter3, valid_iter4, valid_iter5, valid_iter6, 
valid_iter7, valid_iter8, valid_iter9, valid_iter10, valid_iter11, valid_iter12
, valid_iter13, valid_iter14, valid_iter15, valid_iter16, valid_iter17, valid_iter18
, valid_iter19, valid_iter20, valid_iter21, valid_iter22
, valid_iter23, valid_iter24, valid_iter25, valid_iter26, valid_iter27, valid_iter28
, valid_iter29, valid_iter30, valid_iter31, valid_iter32
]

acc = np.empty(shape=(32,)) 
energy = np.linspace(0.125, 7.875, 32)
for i,j in zip(range(32), l):
    acc[i] = valid_ch3(net, j) 
plt.grid(axis='both', linestyle='--', linewidth=1)
plt.xlabel('E (energyFromCharge) / keV')
plt.ylabel('Accuracy')    
plt.plot(energy, acc, lw=2, label='')
plt.show()


