import tensorflow as tf
import random

def create_FCNN(num_layers, worker_idx):
    activations = ['sigmoid','selu','relu',tf.nn.leaky_relu]

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(2, 300),name='input_layer_%d'%worker_idx))
    for i in range(num_layers-1):
        model.add(tf.keras.layers.Dense(random.randint(50,100), activation=random.choice(activations),name='dense%d_%d'%(i,worker_idx)))
    model.add(tf.keras.layers.Dense(2,activation='softmax',name = 'output_layer_%d'%worker_idx))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer,
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

    return model

def explore_FCNN(good_model, bad_model, worker_idx):
    good_layers = good_model.layers
    bad_layers = bad_model.layers
    input_shape = (good_layers[0].input_shape[1],good_layers[0].input_shape[2])

    explore_model = tf.keras.models.Sequential()

    layer_ctr = 1
    for good_l, bad_l in zip(good_layers,bad_layers):
        if 'input' in good_l.name:
            explore_layer = good_l
            explore_model.add(explore_layer)
            layer_ctr += 1
            continue
        elif 'output' in good_l.name:
            explore_output_shape = 2
        else:
            explore_output_shape = max(abs(int(0.8*(good_l.output_shape[1] - bad_l.output_shape[1])) + good_l.output_shape[1]),3)
        # print('Layer-%d, good-%d, bad-%d, explore-%d'%(layer_ctr,good_l.output_shape[1],bad_l.output_shape[1],explore_output_shape))
        
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

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    explore_model.compile(optimizer=optimizer,
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

    return explore_model