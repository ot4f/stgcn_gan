from cgi import test
import numpy as np
import pandas as pd
import torch


def load_pems_data(dataset):
    pems_adj = pd.read_csv(dataset['adj'],header=None)
    W = pems_adj.values
    n = pems_adj.shape[0]
    W = W/10000.
    W2 = W * W
    W_mask = np.ones([n, n]) - np.identity(n)
    w = np.exp(-W2/0.1)
    w = w * (w >= 0.5) * W_mask
    # w = np.mat(w)
    pems_feat = pd.read_csv(dataset['feat'],header=None)
    pems_feat = np.array(pems_feat)
    return pems_feat, w


    
def preprocess_data(data, time_len, rate, seq_len, pre_len, normalize, day_slot=288, n_day=-1, C_0=1):
    x_stats = {}
    if normalize == "minmax":
        maxval = data.max()
        data = data/maxval
        x_stats = {'maxval': maxval}
    n_frame = seq_len + pre_len
    train_data, test_data, val_data = [], [], []
    if n_day != -1:
        n_slot = day_slot - n_frame + 1
        n_route = data.shape[1]
        test_day = int(np.ceil(n_day * rate))
        val_day = test_day
        train_day = n_day - test_day - val_day
        print(train_day, val_day, test_day)
        offset = 0
        for i in range(train_day):
            for j in range(n_slot):
                sta = i*day_slot+j
                a = data[sta:sta+n_frame, :]
                if C_0 > 0:
                    a = np.reshape(data[sta:sta+n_frame, :], (n_frame, n_route, C_0))
                # trainX.append(a[0:seq_len])
                # trainY.append(a[seq_len:n_frame])
                train_data.append(a)
        offset += train_day
        for i in range(val_day):
            for j in range(n_slot):
                sta = (i+offset)*day_slot+j
                a = data[sta:sta+n_frame, :]
                if C_0 > 0:
                    a = np.reshape(data[sta:sta+n_frame, :], (n_frame, n_route, C_0))
                # trainX.append(a[0:seq_len])
                # trainY.append(a[seq_len:n_frame])
                val_data.append(a)
        # test_day = n_day - train_day
        offset += val_day
        for i in range(test_day):
            for j in range(n_slot):
                sta = (i+offset)*day_slot+j
                a = data[sta:sta+n_frame, :]
                if C_0 > 0:
                    a = np.reshape(data[sta:sta+n_frame, :], (n_frame, n_route, C_0))
                test_data.append(a)
    else: 
        train_size = int(time_len*rate)
        train_data_ = data[0:train_size]
        test_data_ = data[train_size:time_len]
        for i in range(len(train_data_) - seq_len - pre_len+1):
            a = train_data_[i: i + seq_len + pre_len]
            # trainX.append(a[0 : seq_len])
            # trainY.append(a[seq_len : seq_len + pre_len])
            train_data.append(a)
        for i in range(len(test_data_) - n_frame+1):
            b = test_data_[i: i + seq_len + pre_len]
            # testX.append(b[0 : seq_len])
            # testY.append(b[seq_len : seq_len + pre_len])
            test_data.append(b)
    
    train_data = np.stack(train_data)
    if len(test_data) > 0:
        test_data = np.stack(test_data)
    if len(val_data) > 0:
        val_data = np.stack(val_data)
    # print(train_data.shape, test_data.shape)
        
    if normalize == 'zscore':
        mean = np.mean(train_data)
        std = np.std(train_data)
        train_data = (train_data - mean) / std
        if len(test_data) > 0:
            test_data = (test_data - mean) / std
        if len(val_data) > 0:
            val_data = (val_data - mean) / std
        x_stats = {'mean': mean, "std": std}
    # import pdb; pdb.set_trace()

    trainX = train_data[:, 0:seq_len]
    trainY = train_data[:, seq_len:n_frame]
    # print(trainX.shape, trainY.shape)
    if len(test_data) > 0:
        testX = test_data[:, 0:seq_len]
        testY = test_data[:, seq_len:n_frame]
    else:
        testX = []
        testY = []

    if len(val_data) > 0:
        valX = val_data[:, 0:seq_len]
        valY = val_data[:, seq_len:n_frame]
    else:
        valX = []
        valY = []
    return trainX, trainY, testX, testY, valX, valY, x_stats
        

def torch_dataset(data, time_len, rate, seq_len, pre_len, normalize='minmax', day_slot=288, n_day=-1, C_0=1):
    trainX, trainY, testX, testY, valX, valY, x_stats = preprocess_data(data, time_len, rate, seq_len, pre_len, normalize, day_slot, n_day, C_0)
    print(trainX.shape ,trainY.shape, testX.shape, testY.shape)
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(trainX), torch.FloatTensor(trainY)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(testX), torch.FloatTensor(testY)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(valX), torch.FloatTensor(valY)
    )
    return train_dataset, test_dataset, val_dataset, x_stats


load_data_mp = {
    'pems': load_pems_data
}