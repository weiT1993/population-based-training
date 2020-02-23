from utils.helper_fun import read_mat
from sklearn.model_selection import train_test_split

def get_dataset(data_file,time_range,concat):
    X, y = read_mat(file_name=data_file,time_range=time_range,concat=concat)
    X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.4, random_state=41)
    dataset_train = (X_train, y_train)

    X_valid, X_test, y_valid, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=41)
    dataset_valid = (X_valid, y_valid)
    dataset_test = (X_test, y_test)
    return dataset_train, dataset_valid, dataset_test