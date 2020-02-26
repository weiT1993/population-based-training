from models import FC
from utils.helper_fun import read_file
import tensorflow as tf

data_dict = read_file(filename='./data/dataset.p')

model = tf.keras.models.load_model('./population/winner_architecture.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
model.summary()
score = model.evaluate(data_dict['valid'][0],data_dict['valid'][1],verbose=1)[1]