import tensorflow as tf
import random
import pickle
import time

class FC():
    def __init__(self, structure, worker_idx):
        self.worker_idx = worker_idx
        if 'layers' not in structure:
            self.model = self.random_model(structure['num_layers'])
        else:
            self.model = self.build_model(structure)
    
    def random_model(self, num_layers):
        activations = [tf.nn.selu,tf.nn.relu,tf.nn.leaky_relu]
        model = tf.keras.models.Sequential(name='worker_%d'%self.worker_idx)
        input_layer = tf.keras.layers.Flatten(input_shape=(300, 2),name='input_layer')
        model.add(input_layer)
        for i in range(num_layers):
            dense_layer = tf.keras.layers.Dense(random.randint(10,600),activation=random.choice(activations),name='dense%d'%i)
            model.add(dense_layer)
        model.add(tf.keras.layers.Dense(2,activation='softmax',name = 'output_layer'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
        self.score = -1
        return model
    
    def build_model(self, structure):
        model = tf.keras.models.Sequential(name='worker_%d'%self.worker_idx)
        input_layer = tf.keras.layers.Flatten(input_shape=(300, 2),name='input_layer')
        model.add(input_layer)
        for i, layer in enumerate(structure['layers']):
            dense_layer = tf.keras.layers.Dense(layer['kernel_size'], activation=layer['activation'],name=layer['name'])
            model.add(dense_layer)
        model.add(tf.keras.layers.Dense(2,activation='softmax',name = 'output_layer'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=structure['lr'])
        model.compile(optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
        self.score = structure['score']
        return model

    def save_model(self):
        layers = self.model.layers
        structure = {}
        structure['layers'] = []
        for layer in layers:
            if 'dense' in layer.name:
                structure['layers'].append({'kernel_size':layer.output_shape[1],
                'activation':layer.activation,
                'name':layer.name})
        structure['lr'] = self.model.optimizer.learning_rate.numpy()
        structure['score'] = self.score
        pickle.dump(structure,open('./population/worker_%d.p'%self.worker_idx,'wb'))
    
    def train(self,dataset_train,dataset_valid):
        x_train, y_train = dataset_train
        x_valid, y_valid = dataset_valid
        self.model.fit(x_train,y_train,epochs=3,verbose=0)
        self.score = self.model.evaluate(x_valid,y_valid,verbose=0)[1]