import os
import tensorflow as tf
import numpy as np
from utils.datasets import get_dataset
from utils.helper_fun import read_file
from models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# data_dict = read_file(filename='./data/dataset.p')
# x_train, y_train = data_dict['train']
# x_valid, y_valid = data_dict['valid']
# x_test, y_test = data_dict['test']

# fc_model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(300, 2),name='input_layer'),
#   tf.keras.layers.Dense(128, activation='relu',name='dense_0'),
#   tf.keras.layers.Dense(130,activation='selu',name = 'dense_1'),
#   tf.keras.layers.Dropout(0.1, name='dp'),
#   tf.keras.layers.Dense(2,activation='softmax',name = 'output_layer')
# ])
# fc_model.summary()
# rand_input = np.array(np.random.rand(1,300,2),dtype='float32')
# predictions = fc_model(rand_input).numpy()
# print(predictions)

# c_model = tf.keras.models.Sequential([
#   tf.keras.layers.Conv1D(filters=16, kernel_size=30, strides=3,activation='relu', input_shape=(300,2)),
#   tf.keras.layers.Conv1D(filters=8, kernel_size=30, activation='relu'),
#   tf.keras.layers.AveragePooling1D(pool_size=10,strides=3),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(2, activation='softmax')
# ])
# c_model.summary()
# rand_input = np.array(np.random.rand(1,300,2),dtype='float32')
# predictions = c_model(rand_input).numpy()
# print(predictions)

# print(c_model.layers[2].strides)

# layers = c_model.layers
# for layer in layers:
# 	if isinstance(layer,tf.keras.layers.Conv1D):
# 		print(layer)

model_0 = Model(worker_idx=0,num_layers=3,model='cnn')
model_0.random_model()

model_1 = Model(worker_idx=1,num_layers=3,model='cnn')
model_1.random_model()
model_1.worker.model.summary()

model_0.explore_model(good_model=model_1.worker.model)
model_0.worker.model.summary()

for layer, perturbed_layer in zip(model_1.worker.model.layers,model_0.worker.model.layers):
    if isinstance(layer,tf.keras.layers.Conv1D):
        print(layer.kernel_size,layer.strides)
        print(perturbed_layer.kernel_size,perturbed_layer.strides)
        print('-'*50)