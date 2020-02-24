import tensorflow as tf
from models import create_FCNN, explore_FCNN
from utils.helper_fun import read_file

data_dict = read_file(filename='./data/dataset.p')
x_train, y_train = data_dict['train']
dataset_valid = data_dict['valid']
x_test, y_test = data_dict['test']

good_model = tf.keras.models.load_model('./checkpoints/generation_0.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu,
        'crelu_v2':tf.nn.crelu})
good_model.fit(x_train, y_train, epochs=10, verbose=2)
good_score = good_model.evaluate(x_test, y_test, verbose=0)[1]

worker_idx = -1
bad_model = create_FCNN(num_layers=3,worker_idx=worker_idx)
bad_score = bad_model.evaluate(x_test, y_test, verbose=0)[1]

good_layers = good_model.layers
bad_layers = bad_model.layers
input_shape = (good_layers[0].input_shape[1],good_layers[0].input_shape[2])

explore_model = explore_FCNN(good_model=good_model,bad_model=bad_model,worker_idx=-1)

explore_score = explore_model.evaluate(x_test, y_test, verbose=0)[1]
print('good score = %.5f, bad score = %.5f, explore score = %.5f'%(good_score,bad_score,explore_score))

explore_pred = explore_model.predict(x_test[:1])
good_pred = good_model.predict(x_test[:1])
print(explore_pred,good_pred)

good_model.summary()
explore_model.summary()