import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn
import numpy as np
import math
import h5py
import pandas as pd
import matplotlib.pyplot as plt


#load the calibration data
hf = h5py.File('../Dongjin/Cast_IAXO_data/calibration-cdl-2018.h5')

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
                'rotationAngle'
                ]
    data_list = [np.array(h5f.get(path + dset)).flatten() for dset in datasets]
    return np.array(data_list).transpose()

calib_dsets_Ag_n = readDatasets(hf, 'calibration-cdl-feb2019-Ag-Ag-6kV/')


# datasets with different energy
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

df = pd.DataFrame(calib_dsets,
                   columns=['eccen',
                   'eFC',
                   'kL','kT','len','sL','sT','frac',#'hits',
                   'rmsL','rmsT'
                   ,'rot'
                   ])

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

#print(dfc_cut)


# background data cut
dfb = pd.DataFrame(background_dsets,
                   columns=['eccen','eFC',
                   'kL','kT','len','sL','sT','frac',#'hits',
                   'rmsL','rmsT'
                   ,'rot'
                   ])

dfb_cut = dfb[ ((dfb['eccen']>0.0) & (dfb['eccen']<10)) & #((dfb['eFC']>0.0) & (dfb['eFC']<5)) &
               ((dfb['kL']>-2) & (dfb['kL']<4)) & ((dfb['kT']>-2) & (dfb['kT']<4)) &
               ((dfb['len']>0) & (dfb['len']<18)) & ((dfb['sL']>-2) & (dfb['sL']<2)) &
               ((dfb['sT']>-2) & (dfb['sT']<2)) & ((dfb['frac']>-1) & (dfb['frac']<1)) &
               #((dfb['hits']>0) & (dfb['hits']<500)) & 
               ((dfb['rmsL']>0) & (dfb['rmsL']<5)) &
               ((dfb['rmsT']>0) & (dfb['rmsT']<2)) ]

dfb_cut = dfb_cut.drop('eFC', axis=1)

# validation data cut

dfv = pd.DataFrame(calib_dsets,
                    columns=['eccen','eFC',
                    'kL','kT','len','sL','sT','frac',#'hits',
                    'rmsL','rmsT'
                    ,'rot'
                    ])

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


# cut for validation and LogL 
def validation_cut(calib_dsets):
    dfv = pd.DataFrame(calib_dsets,
                    columns=['eccen','eFC',
                    'kL','kT','len','sL','sT','frac',#'hits',
                    'rmsL','rmsT'
                    ,'rot'
                    ])

    dfv_cut = dfv[  ((dfv['eccen']>1) & (dfv['eccen']<2.5)) & #((dfv['eFC']>0) & (dfv['eFC']<15)) &
              ((dfv['kL']>-2) & (dfv['kL']<5)) & ((dfv['kT']>-2) & (dfv['kT']<4)) & ((dfv['len']>0) & (dfv['len']<14)) &
              ((dfv['sL']>-2) & (dfv['sL']<2)) & ((dfv['sT']>-2) & (dfv['sT']<2)) & ((dfv['frac']>0) & (dfv['frac']<0.5)) &
              #((df['hits']>0) & (df['hits']<500)) & 
              ((dfv['rmsL']>0) & (dfv['rmsL']<4)) &
              ((dfv['rmsT']>0) & (dfv['rmsT']<2)) & ((dfv['rot']>-0.1) & (dfv['rot']<3.5))  ]

    dfv_cut = dfv_cut.drop('eFC', axis=1)

    return dfv_cut.to_numpy().astype(np.float32)



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

#divide the data for training and test
train_data = calibration_data[:70000]
test_data = calibration_data[70001:80000]

train_data_background = background_data[:70000]
test_data_background = background_data[70001:80000]



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

batch_size = 100

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
 

validation_set1 = Dataset(valid_data, label_calibration)
validation_set2 = Dataset(valid_data_back, label_background)
validation_set = data.ConcatDataset([validation_set1, validation_set2])
valid_iter_cal = data.DataLoader(validation_set1, batch_size, shuffle=True, num_workers = 0)
valid_iter_back = data.DataLoader(validation_set2, batch_size, shuffle=True, num_workers = 0)
valid_iter = data.DataLoader(validation_set, batch_size, shuffle=True, num_workers=0)



#starting mlp

seed = 1
torch.manual_seed(seed)

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

def accuracy(y_hat, y):  #@save
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
        y = y.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):  #@save
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

def accuracy_validation_roc(net, data_iter, set_length):
    if isinstance(net,torch.nn.Module):
        net.eval()
    #tn0 = np.empty(shape=(len(valid_set),))
    #tp1 = np.empty(shape=(len(valid_set),))
    res = np.empty(shape=(set_length,2))
    batch_idx = 0
    correct = 0

    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net.forward(X)  
            pred = y_hat.argmax(1)
            correct += pred.eq(y.argmax(1)).sum()
            #max_value = torch.max(y_hat)
            #tn0[batch_idx * batch_size : (batch_idx + 1) * batch_size] = y_hat[][0] 
            #tp1[batch_idx * batch_size : (batch_idx + 1) * batch_size] = y_hat[][1]
            res[batch_idx * batch_size : (batch_idx + 1) * batch_size, :] = y_hat 
            batch_idx = batch_idx + 1

    return correct/(len(data_iter)*batch_size), res
            


def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """The training loop defined in Chapter 3."""
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

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """Train a model (defined in Chapter 3)."""
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


def valid_ch3(net, valid_iter):
    valid_acc = evaluate_accuracy(net, valid_iter)
    print('Validation Accuracy:', valid_acc[1])
    return valid_acc[1]

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



num_epochs, lr = 100, 0.05
updater = torch.optim.SGD(net.parameters(), lr=lr)

print('check')

train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
valid_ch3(net, valid_iter)
#valid_ch3(net, valid_iter_cal)
#valid_ch3(net, valid_iter_back)

output_cal = valid_roc(net, valid_iter_cal, 20000)
output_back = valid_roc(net, valid_iter_back, 20000)


plt.xlabel('output (neuron 0)')
plt.ylabel('event number #')
plt.hist(output_back[0],100,alpha = 0.5, lw=2, label='background')
plt.hist(output_cal[0],100,alpha = 0.5, lw=2, label='signal_all_calibration')
plt.legend()
plt.show()

plt.xlabel('output (neuron 1)')
plt.ylabel('event number #')
plt.hist(output_back[1],100,alpha = 0.5, lw=2, label= 'background')
plt.hist(output_cal[1],100,alpha = 0.5, lw=2, label= 'signal_all_calibration')
plt.legend()
plt.show()
