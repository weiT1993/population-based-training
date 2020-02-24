import tensorflow as tf
from models import create_FCNN
from utils.helper_fun import read_file

data_dict = read_file(filename='./data/dataset.p')
dataset_train = data_dict['train']
dataset_valid = data_dict['valid']
x_test, y_test = data_dict['test']

good_model = tf.keras.models.load_model('./checkpoints/generation_0.h5')
good_score = good_model.evaluate(x_test, y_test, verbose=0)[1]

worker_idx = -1
bad_model = create_FCNN(num_layers=3,worker_idx=worker_idx)
bad_score = bad_model.evaluate(x_test, y_test, verbose=0)[1]

good_layers = good_model.layers
bad_layers = bad_model.layers
input_shape = (good_layers[0].input_shape[1],good_layers[0].input_shape[2])

explore_model = tf.keras.models.Sequential(tf.keras.layers.Flatten(input_shape=input_shape,name='input_layer_%d'%worker_idx))

layer_ctr = 1
for good_l, bad_l in zip(good_layers[1:-1],bad_layers[1:-1]):
    # print(good_l.name, good_l.activation, bad_l.activation)
    print('bad:',bad_l.input_shape,bad_l.output_shape)
    print('good:',good_l.input_shape,good_l.output_shape)
    explore_output_shape = int(0.8*(good_l.output_shape[1] - bad_l.output_shape[1])) + good_l.output_shape[1]
    explore_layer = tf.keras.layers.Dense(explore_output_shape, 
    activation=good_l.activation,
    name='dense%d_%d'%(layer_ctr,worker_idx),
    kernel_initializer=tf.keras.initializers.Zeros())
    explore_model.add(explore_layer)
    
    good_kernel = good_l.weights[0].numpy()
    explore_kernel = explore_layer.weights[0].numpy()
    for i in range(explore_kernel.shape[0]):
        for j in range(explore_kernel.shape[1]):
            if i < good_kernel.shape[0] and j<good_kernel.shape[1]:
                explore_kernel[i][j] = good_kernel[i][j]
            else:
                explore_kernel[i][j] = 0
    explore_layer.weights[0].assign(explore_kernel)

    good_bias = good_l.weights[1].numpy()
    explore_bias = explore_layer.weights[1].numpy()
    for i in range(explore_bias.shape[0]):
        if i < good_bias.shape[0]:
            explore_bias[i] = good_bias[i]
        else:
            explore_bias[i] = 0
    explore_layer.weights[1].assign(explore_bias)

    layer_ctr += 1

explore_model.add(tf.keras.layers.Dense(2,activation='softmax',name = 'output_layer_%d'%worker_idx))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
explore_model.compile(optimizer=optimizer,
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
explore_score = explore_model.evaluate(x_test, y_test, verbose=0)[1]
print('good score = %.5f, bad score = %.5f, explore score = %.5f'%(good_score,bad_score,explore_score))