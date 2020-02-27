from models import FC
from utils.helper_fun import read_file
from utils.datasets import get_dataset
import tensorflow as tf

data_dict = read_file(filename='./data/dataset.p')
dataset_valid = data_dict['valid']
dataset_test = data_dict['test']

# dataset_train, dataset_valid, dataset_test = get_dataset(data_file='./data/power_15freq_7.3202.mat',time_range=300,concat=False)

model = tf.keras.models.load_model('./3-fc-population/worker_0.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
model.summary()
val_score = model.evaluate(dataset_valid[0],dataset_valid[1],verbose=1)[1]
test_score = model.evaluate(dataset_test[0],dataset_test[1],verbose=1)[1]