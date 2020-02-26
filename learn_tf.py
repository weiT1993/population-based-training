import os
import tensorflow as tf
import numpy as np
from utils.datasets import get_dataset
from utils.helper_fun import read_file
from models import Model

data_dict = read_file(filename='./data/dataset.p')
# x_train, y_train = data_dict['train']
# x_valid, y_valid = data_dict['valid']
# x_test, y_test = data_dict['test']

# fc_model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(300, 2),name='input_layer'),
#   tf.keras.layers.Dense(128, activation='relu',name='dense_0'),
#   tf.keras.layers.Dropout(0.1, name='dp'),
#   tf.keras.layers.Dense(2,activation='softmax',name = 'output_layer')
# ])
# fc_model.summary()
# predictions = fc_model(np.transpose(x_train[:1], (0,2,1))).numpy()
# print(predictions)

# c_model = tf.keras.models.Sequential([
#   tf.keras.layers.Conv1D(filters=32, kernel_size=50, strides=3,activation='relu', input_shape=(300,2)),
#   tf.keras.layers.Conv1D(filters=32, kernel_size=30, activation='relu'),
#   tf.keras.layers.AveragePooling1D(pool_size=30,strides=60),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(2, activation='softmax')
# ])
# c_model.summary()
# predictions = c_model(x_train[:1]).numpy()
# print(predictions)

# layers = c_model.layers
# for layer in layers:
# 	if isinstance(layer,tf.keras.layers.Conv1D):
# 		print(layer)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

worker = Model(worker_idx=0,num_layers=4,model='fc')
worker.load_model(dataset_valid=data_dict['valid'])
print(worker.worker.score)

good_worker = Model(worker_idx=1,num_layers=4,model='fc')
good_worker.load_model(dataset_valid=data_dict['valid'])
print(good_worker.worker.score)

worker.explore_model(good_model=good_worker.worker.model)
print(worker.worker.score)