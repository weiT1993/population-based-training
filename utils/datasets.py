from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
import numpy as np

from utils.helper_fun import read_mat

# def get_dataset(data_file,time_range,concat):
#     X, y = read_mat(file_name=data_file,time_range=time_range,concat=concat)
#     X = np.transpose(X, (0,2,1))
#     X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.4)
#     dataset_train = (X_train, y_train)

#     X_valid, X_test, y_valid, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5)
#     dataset_valid = (X_valid, y_valid)
#     dataset_test = (X_test, y_test)
#     return dataset_train, dataset_valid, dataset_test

# dataset_train, dataset_valid, dataset_test = get_dataset(data_file='./data/power_15freq_7.3202.mat',time_range=300,concat=False)
# pickle.dump({'train':dataset_train,'valid':dataset_valid,'test':dataset_test},open('./data/dataset.p','wb'))

def get_dataset(data_file,dt=8000):
    data = np.load(data_file)
    data = data.astype(np.float)
    [excitedI, excitedQ, groundI, groundQ, groundI_test, groundQ_test] = data

    train_idx = int(len(excitedI)*0.6)
    val_idx = int(len(excitedI)*0.8)

    train_excited = np.concatenate([excitedI[:train_idx,:dt], excitedQ[:train_idx,:dt]], axis=1)
    train_ground = np.concatenate([groundI[:train_idx,:dt], groundQ[:train_idx,:dt]], axis=1)
    train_x = np.concatenate([train_excited, train_ground], axis=0)
    train_y = np.array([1] * len(train_excited) + [0] * len(train_ground))

    val_excited = np.concatenate([excitedI[train_idx:val_idx,:dt], excitedQ[train_idx:val_idx,:dt]], axis=1)
    val_ground = np.concatenate([groundI[train_idx:val_idx,:dt], groundQ[train_idx:val_idx,:dt]], axis=1)
    val_x = np.concatenate([val_excited, val_ground], axis=0)
    val_y = np.array([1] * len(val_excited) + [0] * len(val_ground))

    test_excited = np.concatenate([excitedI[val_idx:,:dt], excitedQ[val_idx:,:dt]], axis=1)
    test_ground = np.concatenate([groundI[val_idx:,:dt], groundQ[val_idx:,:dt]], axis=1)
    test_x = np.concatenate([test_excited, test_ground], axis=0)
    test_y = np.array([1] * len(test_excited) + [0] * len(test_ground))

    scaler = preprocessing.StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    val_x =  scaler.transform(val_x)
    test_x =  scaler.transform(test_x)

    print(test_x.shape)

    dataset_train = (train_x, train_y)    
    dataset_valid = (val_x, val_y)
    dataset_test = (test_x, test_y)
    return dataset_train, dataset_valid, dataset_test

dataset_train, dataset_valid, dataset_test = get_dataset(data_file='./data/power_5freq_7.3233.npy')
# pickle.dump({'train':dataset_train,'valid':dataset_valid,'test':dataset_test},open('./data/dataset.p','wb'))