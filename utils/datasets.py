from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
import numpy as np

from utils.helper_fun import read_mat, read_npy

# def get_dataset(data_file,time_range,concat):
#     X, y = read_mat(file_name=data_file,time_range=time_range,concat=concat)
#     print(X.shape)
#     X = np.transpose(X, (0,2,1))
#     print(X.shape)
#     X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.4)
#     dataset_train = (X_train, y_train)

#     X_valid, X_test, y_valid, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5)
#     dataset_valid = (X_valid, y_valid)
#     dataset_test = (X_test, y_test)

#     print(X_train.shape,y_train.shape)
#     print(X_valid.shape,y_valid.shape)
#     print(X_test.shape,y_test.shape)

#     return dataset_train, dataset_valid, dataset_test

# dataset_train, dataset_valid, dataset_test = get_dataset(data_file='./data/power_15freq_7.3202.mat',time_range=300,concat=False)
# pickle.dump({'train':dataset_train,'valid':dataset_valid,'test':dataset_test},open('./data/dataset.p','wb'))

def get_dataset(data_file,dt=8000):

    X, y = read_npy(file_name=data_file,dt=8000)
    X = np.transpose(X, (0,2,1))

    scaler = preprocessing.StandardScaler().fit(X[:,:,0])
    X[:,:,0] = scaler.transform(X[:,:,0])
    scaler = preprocessing.StandardScaler().fit(X[:,:,1])
    X[:,:,1] = scaler.transform(X[:,:,1])

    X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.4)
    X_valid, X_test, y_valid, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5)

    dataset_train = (X_train, y_train)
    dataset_valid = (X_valid, y_valid)
    dataset_test = (X_test, y_test)

    print(X_train.shape,y_train.shape)
    print(X_valid.shape,y_valid.shape)
    print(X_test.shape,y_test.shape)

    return dataset_train, dataset_valid, dataset_test

dataset_train, dataset_valid, dataset_test = get_dataset(data_file='./data/power_5freq_7.3233.npy')
pickle.dump({'train':dataset_train,'valid':dataset_valid,'test':dataset_test},open('./data/dataset.p','wb'),protocol=4)
