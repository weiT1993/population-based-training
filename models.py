import tensorflow as tf
import random

def create_FCNN(num_layers):
    activations = ['sigmoid','selu']

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(2, 300),name='flatten_input'))
    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(random.randint(50,100), activation=random.choice(activations),name='dense_%d'%i))
    model.add(tf.keras.layers.Dense(2,name = 'output_layer'))

    model.compile(optimizer='adam',
                    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    return model