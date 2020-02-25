import tensorflow as tf
import random
import pickle
import time

class FC():
    def __init__(self, worker_idx):
        self.worker_idx = worker_idx
    
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
        self.model = model
    
    def build_from_structure(self, structure):
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
        self.score = -1
        self.model = model

    def load_model(self,dataset_valid):
        model_h5 = './population/worker_%d.h5'%self.worker_idx
        self.model = tf.keras.models.load_model(model_h5, custom_objects={'leaky_relu': tf.nn.leaky_relu})
        x_valid, y_valid = dataset_valid
        self.score = self.model.evaluate(x_valid,y_valid,verbose=0)[1]
    
    def explore_model(self,good_model,dataset_valid):
        good_layers = good_model.layers
        bad_layers = self.model.layers
        explore_model = tf.keras.models.Sequential(name='worker_%d'%self.worker_idx)
        for good_l, bad_l in zip(good_layers,bad_layers):
            if 'input' in good_l.name:
                explore_layer = good_l
                explore_model.add(explore_layer)
                continue
            elif 'output' in good_l.name:
                explore_kernel_size = 2
            else:
                explore_kernel_size = max(int(0.1*(good_l.output_shape[1] - bad_l.output_shape[1]) + good_l.output_shape[1]),10)
            explore_layer = tf.keras.layers.Dense(explore_kernel_size,activation=good_l.activation)
            explore_model.add(explore_layer)

        explore_model.compile(optimizer=good_model.optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
        self.model = explore_model
        x_valid, y_valid = dataset_valid
        self.score = self.model.evaluate(x_valid,y_valid,verbose=0)[1]

    def save_structure(self):
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

    def save_model(self):
        self.model.save('./population/worker_%d.h5'%self.worker_idx,overwrite=True)
    
    def train(self,dataset_train,dataset_valid):
        x_train, y_train = dataset_train
        x_valid, y_valid = dataset_valid
        self.model.fit(x_train,y_train,epochs=3,verbose=0)
        self.score = self.model.evaluate(x_valid,y_valid,verbose=0)[1]