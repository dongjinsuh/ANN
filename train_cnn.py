import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn
import time 
import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
#TimePix10mu

# load data from h5py file
hf = h5py.File('/home/tpc/data/neural_network/calibration-cdl-2018.h5')
hf_rec = h5py.File('/home/tpc/data/neural_network/reco_186_fixed.h5')

def readDatasets(h5f, path):
    datasets = ['eccentricity',
                #'energyFromCharge',
                'kurtosisLongitudinal',
                'kurtosisTransverse',
                'length',
                'skewnessLongitudinal',
                'skewnessTransverse',
                'fractionInTransverseRms',
                #'hits',
                'rmsLongitudinal',
                'rmsTransverse',
                'rotationAngle']
    data_list = [np.array(h5f.get(path + dset)).flatten() for dset in datasets]
    return np.array(data_list).transpose()

calib_dsets_Ag = readDatasets(hf, 'calibration-cdl-feb2019-Ag-Ag-6kV/')

calib_dsets_Cu2 = readDatasets(hf, 'calibration-cdl-feb2019-Cu-EPIC-2kV/')
calib_dsets_Cu09 = readDatasets(hf, 'calibration-cdl-feb2019-Cu-EPIC-0.9kV/')
calib_dsets_Al = readDatasets(hf, 'calibration-cdl-feb2019-Al-Al-4kV/')
calib_dsets_C = readDatasets(hf, 'calibration-cdl-feb2019-C-EPIC-0.6kV/')
calib_dsets_Cu_Ni = readDatasets(hf, 'calibration-cdl-feb2019-Cu-Ni-15kV/')
calib_dsets_Mn_Cr = readDatasets(hf, 'calibration-cdl-feb2019-Mn-Cr-12kV/')
calib_dsets_Ti = readDatasets(hf, 'calibration-cdl-feb2019-Ti-Ti-9kV/')
background_dsets = readDatasets(hf_rec, 'reconstruction/run_186/chip_3/')


def readxyDatasets(h5f, path):
    data = np.array(h5f.get(path))
    return data


b_x = readxyDatasets(hf_rec, 'reconstruction/run_186/chip_3/x')
b_y =readxyDatasets(hf_rec, 'reconstruction/run_186/chip_3/y')
b_charge = readxyDatasets(hf_rec, 'reconstruction/run_186/chip_3/charge')


ag_x = readxyDatasets(hf, 'calibration-cdl-feb2019-Ag-Ag-6kV/x')
ag_y = readxyDatasets(hf, 'calibration-cdl-feb2019-Ag-Ag-6kV/y')
ag_charge = readxyDatasets(hf, 'calibration-cdl-feb2019-Ag-Ag-6kV/charge')


al_x = readxyDatasets(hf, 'calibration-cdl-feb2019-Al-Al-4kV/x')
al_y = readxyDatasets(hf, 'calibration-cdl-feb2019-Al-Al-4kV/y')
al_charge = readxyDatasets(hf, 'calibration-cdl-feb2019-Al-Al-4kV/charge')

c_x = readxyDatasets(hf, 'calibration-cdl-feb2019-C-EPIC-0.6kV/x')
c_y = readxyDatasets(hf, 'calibration-cdl-feb2019-C-EPIC-0.6kV/y')
c_charge = readxyDatasets(hf, 'calibration-cdl-feb2019-C-EPIC-0.6kV/charge')

cu1_x = readxyDatasets(hf, 'calibration-cdl-feb2019-Cu-EPIC-0.9kV/x')
cu1_y = readxyDatasets(hf, 'calibration-cdl-feb2019-Cu-EPIC-0.9kV/y')
cu1_charge = readxyDatasets(hf, 'calibration-cdl-feb2019-Cu-EPIC-0.9kV/charge')

cu2_x = readxyDatasets(hf, 'calibration-cdl-feb2019-Cu-EPIC-2kV/x')
cu2_y = readxyDatasets(hf, 'calibration-cdl-feb2019-Cu-EPIC-2kV/y')
cu2_charge = readxyDatasets(hf, 'calibration-cdl-feb2019-Cu-EPIC-2kV/charge')

cn_x = readxyDatasets(hf, 'calibration-cdl-feb2019-Cu-Ni-15kV/x')
cn_y = readxyDatasets(hf, 'calibration-cdl-feb2019-Cu-Ni-15kV/y')
cn_charge = readxyDatasets(hf, 'calibration-cdl-feb2019-Cu-Ni-15kV/charge')

mc_x = readxyDatasets(hf, 'calibration-cdl-feb2019-Mn-Cr-12kV/x')
mc_y = readxyDatasets(hf, 'calibration-cdl-feb2019-Mn-Cr-12kV/y')
mc_charge = readxyDatasets(hf, 'calibration-cdl-feb2019-Mn-Cr-12kV/charge')

ti_x = readxyDatasets(hf, 'calibration-cdl-feb2019-Ti-Ti-9kV/x')
ti_y = readxyDatasets(hf, 'calibration-cdl-feb2019-Ti-Ti-9kV/y')
ti_charge = readxyDatasets(hf, 'calibration-cdl-feb2019-Ti-Ti-9kV/charge')

hf.close
hf_rec.close


calib_dsets = np.concatenate((calib_dsets_Ag, calib_dsets_Al, calib_dsets_C, 
                            calib_dsets_Cu09 
                            ,calib_dsets_Cu2
                            , calib_dsets_Cu_Ni
                            , calib_dsets_Mn_Cr
                            , calib_dsets_Ti),axis=0)
x_dsets = np.concatenate((ag_x, al_x, c_x, cu1_x,cu2_x, cn_x,mc_x, ti_x),axis=0) 
y_dsets = np.concatenate((ag_y, al_y, c_y, cu1_y, cu2_y, cn_y, mc_y, ti_y),axis=0)
charge_dsets = np.concatenate((ag_charge, al_charge,c_charge, cu1_charge, cu2_charge, cn_charge, mc_charge, ti_charge),axis=0)


# data cut for calibration data
def datacut(calib_dsets):
    df = pd.DataFrame(calib_dsets,
                   columns=['eccen',
                   #'eFC',
                   'kL','kT','len','sL','sT','frac',#'hits',
                   'rmsL','rmsT'
                   ,'rot'
                   #,'likeleíhood'
                   ])
    dfc_cut = df[ ((df['eccen']>1) & (df['eccen']<5)) & #((df['eFC']>0) & (df['eFC']<15)) &
                   ((df['kL']>-2) & (df['kL']<5)) & ((df['kT']>-2) & (df['kT']<4)) & ((df['len']>0) & (df['len']<14)) &
                   ((df['sL']>-2) & (df['sL']<2)) & ((df['sT']>-2) & (df['sT']<2)) & ((df['frac']>0) & (df['frac']<0.5)) &
                  #((df['hits']>0) & (df['hits']<500)) & 
                   ((df['rmsL']>0) & (df['rmsL']<4)) &
                   ((df['rmsT']>0) & (df['rmsT']<2)) & ((df['rot']>-0.1) & (df['rot']<3.5)) ]
    b = dfc_cut.index.tolist()
    return b 


 # preparing dataset for cnn
num_data_cal = 16000 #len(c_charge)
num_data_back = 21001 #len(b_charge)

dataset_cal = np.empty(shape=(num_data_cal,1,256,256),dtype=np.float32)
dataset_background = np.empty(shape=(num_data_back,1,256,256),dtype=np.float32)

def setupCalibration(b, xdata, ydata, chargedata):
    for j,n in zip(b, range(num_data_cal)):
        charge = chargedata[j][0]
        x = xdata[j][0]
        y = ydata[j][0]
        for i in range(len(charge)):
            dataset_cal[n][0][x[i]][y[i]] = np.float32(charge[i])
        if n == num_data_cal:
            break 
    return dataset_cal

dataset_calibration = setupCalibration(datacut(calib_dsets), x_dsets, y_dsets, charge_dsets)

for j in range(num_data_back):
    charge = b_charge[j+5000][0]
    x = b_x[j+5000][0]
    y = b_y[j+5000][0]
    for i in range(len(charge)):
        dataset_background[j][0][x[i]][y[i]] = charge[i]

dataset_calibration = dataset_calibration.astype(np.float32)
dataset_background = dataset_background.astype(np.float32)



#shuffle the data random
seed = 10
np.random.seed(seed)
np.random.shuffle(dataset_calibration)
np.random.shuffle(dataset_background)

#divide the data for training and test 
train_data = dataset_calibration[:15000]
test_data = dataset_calibration[15001:16000]

train_data_background = dataset_background[:15000]
test_data_background = dataset_background[15001:16000]


# setup validation data

num_data_val = 5000
def setupValidation(dset, xdata, ydata, chargedata):
    dataset_val = np.empty(shape=(num_data_val,1,256,256),dtype=np.float32)
    dfv = pd.DataFrame(dset,
                   columns=['eccen',#'eFC',
                   'kL','kT','len','sL','sT','frac',#'hits',
                   'rmsL','rmsT'
                   ,'rot'])

    dfv_cut = dfv[ ((dfv['eccen']>0.0) & (dfv['eccen']<5)) &# ((dfv['eFC']>0.0) & (dfv['eFC']<15)) &
               ((dfv['kL']>-2) & (dfv['kL']<5)) & ((dfv['kT']>-2) & (dfv['kT']<4)) &
               ((dfv['len']>0) & (dfv['len']<18)) & ((dfv['sL']>-2) & (dfv['sL']<2)) &
               ((dfv['sT']>-2) & (dfv['sT']<2)) & ((dfv['frac']>0) & (dfv['frac']<1)) &
               #((dfv['hits']>0) & (dfv['hits']<500)) & 
               ((dfv['rmsL']>0) & (dfv['rmsL']<5)) &
               ((dfv['rmsT']>0) & (dfv['rmsT']<2)) ]
    v = dfv_cut.index.tolist()

    for j,n in zip(v, range(num_data_val)):
        charge = chargedata[j][0]
        x = xdata[j][0]
        y = ydata[j][0]
        for i in range(len(charge)):
            dataset_val[n][0][x[i]][y[i]] = charge[i] 
        if n == num_data_val:
            break
    return dataset_val


dataset_validation = setupValidation(calib_dsets, x_dsets, y_dsets, charge_dsets) 
np.random.shuffle(dataset_validation)

dataset_validation = dataset_validation.astype(np.float32)
valid_data_cal = dataset_validation[:5000]
valid_data_back = dataset_background[1600:21000]


 # get label to dataset
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


valid_set_cal = Dataset(valid_data_cal, label_calibration)
valid_set_back = Dataset(valid_data_back, label_background)
valid_set = data.ConcatDataset([valid_set_cal, valid_set_back])
valid_iter_cal = data.DataLoader(valid_set_cal, batch_size, shuffle=True, num_workers = 0)
valid_iter_back = data.DataLoader(valid_set_back, batch_size, shuffle=True, num_workers = 0)
valid_iter = data.DataLoader(valid_set, batch_size, shuffle=True, num_workers = 0)


# define some necessary classes and functions

def get_dataloader_workers():  #@save
    """Use # processes to read the data."""
    return 0


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



def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().
    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')



####
 # start cnn
seed = 1
torch.manual_seed(seed)

net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), 
                    nn.Conv2d(1, 1, kernel_size=5, padding=2), nn.Tanh(), 
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(1, 1, kernel_size=5, padding=2), nn.Tanh(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(1, 1, kernel_size=5, padding=2), nn.Tanh(), 
                    #nn.MaxPool2d(kernel_size=2, stride=2),
                    #nn.Conv2d(1, 1, kernel_size=5), nn.Tanh(),
                    nn.Flatten(),
                    nn.Linear(1024, 300), nn.Tanh(),
                    #nn.Linear(144, 30), nn.Tanh(),
                    nn.Linear(300, 30), nn.Tanh(), 
                    nn.Linear(30, 2))


def evaluate_accuracy_gpu(net, data_iter, device=None):  #@save
    """Compute the accuracy for a model on a dataset using a GPU."""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)
    correct = 0 

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
            y_hat = net(X)
            pred = y_hat.argmax(1)
            correct += pred.eq(y.argmax(1)).sum()
    return metric[0] / metric[1], correct / (len(data_iter)*batch_size)


def accuracy_validation_roc(net, data_iter, set_length, device=None):  #@save
    """Compute the accuracy for a model on a dataset using a GPU."""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    res = np.empty(shape=(set_length,2))
    batch_idx = 0
    correct = 0 

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            y_hat_np = y_hat.detach().cpu().numpy()
            pred = y_hat.argmax(1)
            correct += pred.eq(y.argmax(1)).sum()

            res[batch_idx * batch_size : (batch_idx + 1) * batch_size, :] = y_hat_np
            batch_idx = batch_idx + 1

    return  correct / (len(data_iter)*batch_size), res



def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6)."""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    x = np.arange(num_epochs)
    y_loss = np.empty(shape=(num_epochs,))
    y_train_acc = np.empty(shape=(num_epochs,))
    y_test_acc = np.empty(shape=(num_epochs,))

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    #loss = nn.CrossEntropyLoss()
    loss = nn.BCEWithLogitsLoss()
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(3)
        net.train()
        correct = 0
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            pred = y_hat.argmax(1)
            correct += pred.eq(y.argmax(1)).sum() 

        train_acc = correct / (len(train_iter)*batch_size)
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print('Epoch',epoch,':  ' 'train_loss=',train_l, 'train_acc=',train_acc, 'test_acc=',test_acc[1])
        y_loss[epoch] = train_l 
        y_train_acc[epoch] = train_acc 
        y_test_acc[epoch] = test_acc[1]

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc[1]:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    fig, ax = plt.subplots(1, figsize = (10,10))
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    #plt.plot(x, y_loss, lw=2, label='loss')
    plt.plot(x, y_train_acc, 'r', lw=2, label='train_acc')
    plt.plot(x, y_test_acc, 'b', lw=2, label='test_acc')
    plt.legend()
    #plt.savefig('cnn_acc_xx')
    plt.show()


def valid_ch6(net, valid_iter):
    valid_acc = evaluate_accuracy_gpu(net, valid_iter)
    print('Validation Accuracy:', valid_acc[1])
    return 0


def valid_roc(net, valid_iter, set_length):
    valid_acc = accuracy_validation_roc(net, valid_iter, set_length)
    print('Validation accuracy:', valid_acc[0])
    res = valid_acc[1]
    res0 = res[:,0]
    res1 = res[:,1]
    return res0, res1


#calculate roc-curve
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
    #print(roc[400:600,0])

    # if signal is lower, roc[0] is signal efficiency
    return roc[:,0], roc[:,1] 
'''
    plt.xlabel('signal efficiency')
    plt.ylabel('background rejection')
    plt.plot(roc[:,1], roc[:,0],lw=2,label='')
    #plt.legend()
    plt.show()
'''


lr, num_epochs = 0.01, 100

print('check')

train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())
valid_ch6(net, valid_iter)

num_vset = num_data_val

output_cal = valid_roc(net, valid_iter_cal, num_vset)
output_back = valid_roc(net, valid_iter_back, num_vset)

plt.figure()
plt.xlabel('output (neuron 0)')
plt.ylabel('event number #')
plt.hist(output_back[0],100,alpha = 0.5, lw=2, label='background')
plt.hist(output_cal[0],100,alpha = 0.5, lw=2, label='signal_calibration all')
plt.legend()
plt.savefig('output0_valid_all_cnn')
plt.show()

plt.figure()
plt.xlabel('output (neuron 1)')
plt.ylabel('event number #')
plt.hist(output_back[1],100,alpha = 0.5, lw=2, label= 'background')
plt.hist(output_cal[1],100,alpha = 0.5, lw=2, label= 'signal_calibration all')
plt.legend()
plt.savefig('output1_valid_all_cnn')
#plt.savefig('output1_distribution_all_100epoch_cnn_valid_new')
plt.show()


