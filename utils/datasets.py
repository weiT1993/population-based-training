from utils.helper_fun import read_mat
from sklearn.model_selection import train_test_split

def get_dataset(data_file,time_range,concat):
    X, y = read_mat(file_name=data_file,time_range=time_range,concat=concat)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)
    dataset_train = (X_train, y_train)
    dataset_valid = (X_valid, y_valid)
    return dataset_train, dataset_valid