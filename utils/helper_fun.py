import numpy as np
import math
from scipy.io import loadmat
import csv
import os
import pickle

def read_mat(file_name, time_range, concat):
    # print("Loading %s" % file_name)
    f = loadmat(file_name)
    ground_I = np.array(f.get('groundI'))[:,310:min(310+time_range, 2048)]
    ground_Q = np.array(f.get('groundQ'))[:,310:min(310+time_range, 2048)]
    excited_I = np.array(f.get('excitedI'))[:,310:min(310+time_range, 2048)]
    excited_Q = np.array(f.get('excitedQ'))[:,310:min(310+time_range, 2048)]

    if concat:
        ground_x = np.hstack((ground_I, ground_Q))
        ground_y = np.zeros(len(ground_x), dtype=np.int)
        excited_x = np.hstack((excited_I, excited_Q))
        excited_y = np.ones(len(excited_x), dtype=np.int)

        X = np.vstack((ground_x, excited_x))
        y = np.hstack((ground_y, excited_y))

        return X, y
    else:
        ground_x = []
        for sample_I, sample_Q in zip(ground_I, ground_Q):
            ground_x.append([sample_I,sample_Q])
        ground_x = np.array(ground_x)
        ground_y = np.zeros(ground_x.shape[0], dtype=np.int)

        excited_x = []
        for sample_I, sample_Q in zip(excited_I, excited_Q):
            excited_x.append([sample_I,sample_Q])
        excited_x = np.array(excited_x)
        excited_y = np.ones(excited_x.shape[0], dtype=np.int)

        X = np.vstack((ground_x, excited_x))
        y = np.hstack((ground_y, excited_y))
        return X, y

def read_npy(file_name,dt):
    data = np.load(file_name)
    data = data.astype(np.float)
    [excited_I, excited_Q, ground_I, ground_Q, groundI_test, groundQ_test] = data

    ground_x = []
    for sample_I, sample_Q in zip(ground_I[:,:dt], ground_Q[:,:dt]):
        ground_x.append([sample_I,sample_Q])
    ground_x = np.array(ground_x)
    ground_y = np.zeros(ground_x.shape[0], dtype=np.int)

    excited_x = []
    for sample_I, sample_Q in zip(excited_I[:,:dt], excited_Q[:,:dt]):
        excited_x.append([sample_I,sample_Q])
    excited_x = np.array(excited_x)
    excited_y = np.ones(excited_x.shape[0], dtype=np.int)

    X = np.vstack((ground_x, excited_x))
    y = np.hstack((ground_y, excited_y))
    return X, y

def read_file(filename):
    if os.path.isfile(filename):
        f = open(filename,'rb')
        file_content = {}
        while 1:
            try:
                file_content.update(pickle.load(f))
            except (EOFError):
                break
        f.close()
    else:
        file_content = {}
    return file_content

def convertfile(path_to_file,train,timestep):
    mat = loadmat(path_to_file)
    mat_shape = mat['excitedI'].shape
    timestep = mat_shape[1] if timestep==0 else timestep
    assert timestep<=mat_shape[1]
    train_idx = np.random.choice(mat_shape[0], math.ceil(mat_shape[0]*train), replace=False)
    test_idx = list(set(range(mat_shape[0])) - set(train_idx))

    excitedI_train = mat['excitedI'][train_idx,:timestep]
    excitedI_test = mat['excitedI'][test_idx,:timestep]
    excitedQ_train = mat['excitedQ'][train_idx,:timestep]
    excitedQ_test = mat['excitedQ'][test_idx,:timestep]
    excited_train = np.concatenate((excitedI_train,excitedQ_train),axis=1)
    excited_train_labels = np.ones((excited_train.shape[0],1))
    excited_train = np.concatenate((excited_train,excited_train_labels),axis=1)
    excited_test = np.concatenate((excitedI_test,excitedQ_test),axis=1)
    excited_test_labels = np.ones((excited_test.shape[0],1))
    excited_test = np.concatenate((excited_test,excited_test_labels),axis=1)

    groundI_train = mat['groundI'][train_idx,:timestep]
    groundI_test = mat['groundI'][test_idx,:timestep]
    groundQ_train = mat['groundQ'][train_idx,:timestep]
    groundQ_test = mat['groundQ'][test_idx,:timestep]
    ground_train = np.concatenate((groundI_train,groundQ_train),axis=1)
    ground_train_labels = np.zeros((ground_train.shape[0],1))
    ground_train = np.concatenate((ground_train,ground_train_labels),axis=1)
    ground_test = np.concatenate((groundI_test,groundQ_test),axis=1)
    ground_test_labels = np.zeros((ground_test.shape[0],1))
    ground_test = np.concatenate((ground_test,ground_test_labels),axis=1)

    train = np.concatenate((ground_train,excited_train),axis=0)
    test = np.concatenate((ground_test,excited_test),axis=0)
    np.random.shuffle(train)
    np.random.shuffle(test)

    dirname = os.path.dirname(path_to_file)
    filename = os.path.basename(path_to_file)
    basename, file_extention = os.path.splitext(filename)
    data = {'train':train,'test':test}
    csv_path_to_files = {}
    for key in data:
        csv_path_to_file = './data/%s_%s.csv'%(basename,key)
        csv_path_to_files[key] = csv_path_to_file
        with open(csv_path_to_file, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['I_%d'%x for x in range(timestep)]+['Q_%d'%x for x in range(timestep)]+['class'])
            csv_writer.writerows(row for row in data[key])
    return csv_path_to_files

def perturb(a,coefficient,force_int):
    if a!=0:
        perturbed = abs(np.random.normal(a,a/coefficient))
    else:
        perturbed = abs(np.random.normal(0,1))
    if force_int:
        return math.ceil(perturbed)
    else:
        return perturbed