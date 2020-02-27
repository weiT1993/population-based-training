import os
import tensorflow as tf
import numpy as np
from utils.datasets import get_dataset
from utils.helper_fun import read_file
from models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_dict = read_file(filename='./data/dataset.p')
x_train, y_train = data_dict['train']
x_valid, y_valid = data_dict['valid']
x_test, y_test = data_dict['test']

# dataset_train, dataset_valid, dataset_test = get_dataset(data_file='./data/power_15freq_7.3202.mat',time_range=300,concat=False)

rnn_model = tf.keras.Sequential()
rnn_model.add(tf.keras.layers.Embedding(input_dim=1000, output_dim=64))
rnn_model.add(tf.keras.layers.LSTM(128))
rnn_model.add(tf.keras.layers.Dense(2))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
rnn_model.compile(optimizer=optimizer,
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])
rnn_model.summary()

rnn_model.fit(x_train, y_train,
          validation_data=(x_valid, y_valid),
          epochs=5)
rnn_model.evaluate(x_test,y_test,verbose=1)