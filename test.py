from datetime import datetime
from models import FC
from utils.helper_fun import read_file
from utils.datasets import get_dataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

#data_dict = read_file(filename='./data/dataset.p')
#dataset_train = data_dict['train']
#dataset_valid = data_dict['valid']
#dataset_test = data_dict['test']

dataset_train, dataset_valid, dataset_test = get_dataset(data_file='./data/power_15freq_7.3202.mat',time_range=300,concat=False)

model = tf.keras.models.load_model('./3-fc-population/worker_4.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
model.summary()
print(model.optimizer.learning_rate)

logdir='./logs/'+datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

model.fit(dataset_train[0],dataset_train[1],epochs=5,verbose=1,callbacks=[tensorboard_callback])
val_score = model.evaluate(dataset_valid[0],dataset_valid[1],verbose=1)[1]
test_score = model.evaluate(dataset_test[0],dataset_test[1],verbose=1)[1]
print('val score = %.5f, test score = %.5f'%(val_score, test_score))