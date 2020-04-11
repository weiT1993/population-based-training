import os
import tensorflow as tf
import random
import pickle
import time
import numpy as np
from utils.helper_fun import perturb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Model():
    def __init__(self, worker_idx, num_layers, model):
        if model == 'fc':
            self.worker = FC(worker_idx=worker_idx,num_layers=num_layers)
        elif model == 'cnn':
            self.worker = CNN(worker_idx=worker_idx,num_layers=num_layers)
        else:
            raise Exception('Model %s is not implemented'%model)
    
    def random_model(self):
        self.worker.random_model()
    
    def load_model(self,dataset_valid):
        self.worker.load_model(dataset_valid=dataset_valid)

    def explore_model(self,good_model):
        self.worker.explore_model(good_model=good_model)
    
    def save_model(self,save_mode):
        self.worker.save_model(save_mode=save_mode)
    
    def train(self,dataset_train,dataset_valid):
        self.worker.train(dataset_train=dataset_train,dataset_valid=dataset_valid)

class FC():
    def __init__(self, worker_idx, num_layers):
        self.worker_idx = worker_idx
        self.num_layers = num_layers
    
    def random_model(self):
        activations = [tf.nn.selu,tf.nn.relu,tf.nn.leaky_relu]
        learning_rates = [0.05,0.01,0.005,0.001]
        l2_norms = [0.01,0.001,0.0001]
        dropout_rates = [0.5,0.4,0.3,0.2,0.1]
        
        model = tf.keras.models.Sequential(name='worker_%d'%self.worker_idx)
        input_layer = tf.keras.layers.Flatten(input_shape=(4096, 2),name='input_layer')
        model.add(input_layer)
        for i in range(self.num_layers):
            dense_layer = tf.keras.layers.Dense(random.randint(10,1000),
            activation=random.choice(activations),
            kernel_regularizer=tf.keras.regularizers.l2(random.choice(l2_norms)))
            model.add(dense_layer)
            dropout_layer = tf.keras.layers.Dropout(random.choice(dropout_rates))
            model.add(dropout_layer)
        model.add(tf.keras.layers.Dense(2,activation='softmax',name = 'output_layer'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=random.choice(learning_rates))
        model.compile(optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
        self.score = -1
        self.model = model

    def load_model(self,dataset_valid):
        model_h5 = './%d-fc-population/worker_%d.h5'%(self.num_layers,self.worker_idx)
        self.model = tf.keras.models.load_model(model_h5, custom_objects={'leaky_relu': tf.nn.leaky_relu})
        x_valid, y_valid = dataset_valid
        self.score = self.model.evaluate(x_valid,y_valid,verbose=0)[1]
    
    def explore_model(self,good_model):
        activations = [tf.nn.selu,tf.nn.relu,tf.nn.leaky_relu]
        good_layers = good_model.layers
        explore_model = tf.keras.models.Sequential(name='worker_%d'%self.worker_idx)
        input_layer = tf.keras.layers.Flatten(input_shape=(4096, 2),name='input_layer')
        explore_model.add(input_layer)
        for good_l in good_layers:
            if 'input' in good_l.name or 'output' in good_l.name:
                continue
            elif isinstance(good_l,tf.keras.layers.Dropout):
                explore_dropout_rate = perturb(a=good_l.rate,coefficient=10,force_int=False)
                explore_dropout_layer = tf.keras.layers.Dropout(explore_dropout_rate)
                explore_model.add(explore_dropout_layer)
            else:
                explore_units_size = perturb(a=good_l.units,coefficient=3,force_int=True)
                explore_l2_norm = perturb(a=good_l.kernel_regularizer.l2,coefficient=10,force_int=False)
                explore_dense_layer = tf.keras.layers.Dense(explore_units_size,activation=random.choice(activations),
                kernel_regularizer=tf.keras.regularizers.l2(explore_l2_norm))
                explore_model.add(explore_dense_layer)
        output_layer = tf.keras.layers.Dense(2,activation='softmax',name = 'output_layer')
        explore_model.add(output_layer)

        good_optimizer_lr = good_model.optimizer.learning_rate.numpy()
        explore_optimizer_lr = perturb(a=good_optimizer_lr,coefficient=5,force_int=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=explore_optimizer_lr)
        explore_model.compile(optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
        self.model = explore_model
        self.score = -1

    def save_model(self, save_mode):
        if save_mode == 0:
            self.model.save('./%d-fc-population/worker_%d.h5'%(self.num_layers,self.worker_idx),overwrite=True)
        elif save_mode == 1:
            self.model.save('./%d-fc-population/best_architecture.h5'%self.num_layers,overwrite=True)
        elif save_mode == 2:
            self.model.save('./%d-fc-population/winner_architecture.h5'%self.num_layers,overwrite=True)
        else:
            raise Exception('Illegal save_mode = %d'%save_mode)
    
    def train(self,dataset_train,dataset_valid):
        x_train, y_train = dataset_train
        x_valid, y_valid = dataset_valid
        self.model.fit(x_train,y_train,epochs=3,verbose=0)
        self.score = self.model.evaluate(x_valid,y_valid,verbose=0)[1]

class CNN():
    def __init__(self, worker_idx, num_layers):
        self.worker_idx = worker_idx
        self.num_layers = num_layers
        self.activations = [tf.nn.selu,tf.nn.relu,tf.nn.leaky_relu]
        self.learning_rates = [0.1,0.05,0.01,0.005,0.001]
        self.max_filters = 50
        self.max_kernel_size = 200
        self.max_strides = 3
    
    def random_model(self):
        pool_layer_idxs = random.sample(range(self.num_layers),random.randint(1,int(self.num_layers/3)))
        l2_norms = [0.01,0.001,0.0001]
        
        model = tf.keras.models.Sequential(name='worker_%d'%self.worker_idx)
        input_layer = tf.keras.layers.Conv1D(filters=random.randint(2,self.max_filters),
        kernel_size=random.randint(1,self.max_kernel_size),
        strides=random.randint(1,self.max_strides),
        activation=random.choice(self.activations),
        kernel_regularizer=tf.keras.regularizers.l2(random.choice(l2_norms)),
        padding='causal',
        input_shape=(4096,2))
        model.add(input_layer)

        for i in range(self.num_layers):
            # print('Adding layer-%d'%i)
            max_kernel_size = model.layers[-1].output_shape[1]
            conv_layer = tf.keras.layers.Conv1D(filters=random.randint(2,self.max_filters),
            kernel_size=random.randint(1,max_kernel_size),
            strides=random.randint(1,self.max_strides),
            activation=random.choice(self.activations),
            kernel_regularizer=tf.keras.regularizers.l2(random.choice(l2_norms)),
            padding='causal')
            model.add(conv_layer)
            # model.summary()
            if i in pool_layer_idxs:
                max_pool_size = model.layers[-1].output_shape[1]
                # print(max_pool_size)
                pool_layer = tf.keras.layers.AveragePooling1D(pool_size=random.randint(2,max_pool_size),strides=random.randint(1,self.max_strides))
                model.add(pool_layer)
                # model.summary()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(2,activation='softmax',name = 'output_layer'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=random.choice(self.learning_rates))
        model.compile(optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
        self.score = -1
        self.model = model
        # model.summary()

    def load_model(self,dataset_valid):
        model_h5 = './%d-cnn-population/worker_%d.h5'%(self.num_layers,self.worker_idx)
        self.model = tf.keras.models.load_model(model_h5, custom_objects={'leaky_relu': tf.nn.leaky_relu})
        x_valid, y_valid = dataset_valid
        self.score = self.model.evaluate(x_valid,y_valid,verbose=0)[1]
    
    def explore_model(self,good_model):
        good_layers = good_model.layers
        bad_conv_layers = []
        for layer in self.model.layers:
            if isinstance(layer,tf.keras.layers.Conv1D):
                bad_conv_layers.append(layer)
        
        explore_model = tf.keras.models.Sequential(name='worker_%d'%self.worker_idx)

        for layer_ctr, good_l in enumerate(good_layers):
            if isinstance(good_l,tf.keras.layers.Conv1D) and layer_ctr==0:

                explore_filters = perturb(a=good_l.filters,coefficient=3,force_int=True)
                explore_kernel_size = perturb(a=good_l.kernel_size[0],coefficient=3,force_int=True)
                explore_kernel_size = min(300,explore_kernel_size)
                explore_strides = perturb(a=good_l.strides[0],coefficient=3,force_int=True)
                explore_l2_norm = perturb(a=good_l.kernel_regularizer.l2,coefficient=10,force_int=False)
                
                input_layer = tf.keras.layers.Conv1D(filters=explore_filters,
                kernel_size=explore_kernel_size,
                strides=explore_strides,
                activation=random.choice(self.activations),
                kernel_regularizer=tf.keras.regularizers.l2(explore_l2_norm),
                padding='causal',
                input_shape=(4096,2))
                explore_model.add(input_layer)
            elif isinstance(good_l,tf.keras.layers.Conv1D):
                max_kernel_size = explore_model.layers[-1].output_shape[1]

                explore_filters = perturb(a=good_l.filters,coefficient=3,force_int=True)
                explore_kernel_size = perturb(a=good_l.kernel_size[0],coefficient=3,force_int=True)
                explore_kernel_size = min(max_kernel_size,explore_kernel_size)
                explore_strides = perturb(a=good_l.strides[0],coefficient=3,force_int=True)
                explore_l2_norm = perturb(a=good_l.kernel_regularizer.l2,coefficient=10,force_int=False)

                conv_layer = tf.keras.layers.Conv1D(filters=explore_filters,
                kernel_size=explore_kernel_size,
                strides=explore_strides,
                activation=random.choice(self.activations),
                kernel_regularizer=tf.keras.regularizers.l2(explore_l2_norm),
                padding='causal')
                explore_model.add(conv_layer)
            elif isinstance(good_l,tf.keras.layers.AveragePooling1D):
                max_pool_size = explore_model.layers[-1].output_shape[1]
                explore_pool_size = perturb(a=good_l.pool_size[0],coefficient=3,force_int=True)
                explore_pool_size = min(max_pool_size,explore_pool_size)
                explore_strides = perturb(a=good_l.strides[0],coefficient=3,force_int=True)
                pool_layer = tf.keras.layers.AveragePooling1D(pool_size=explore_pool_size,strides=explore_strides)
                explore_model.add(pool_layer)
            else:
                continue
        
        explore_model.add(tf.keras.layers.Flatten())
        explore_model.add(tf.keras.layers.Dense(2,activation='softmax',name = 'output_layer'))

        good_optimizer_lr = good_model.optimizer.learning_rate.numpy()
        explore_optimizer_lr = perturb(a=good_optimizer_lr,coefficient=5,force_int=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=explore_optimizer_lr)
        explore_model.compile(optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
        self.model = explore_model
        self.score = -1

    def save_model(self, save_mode):
        if save_mode == 0:
            self.model.save('./%d-cnn-population/worker_%d.h5'%(self.num_layers,self.worker_idx),overwrite=True)
        elif save_mode == 1:
            self.model.save('./%d-cnn-population/best_architecture.h5'%self.num_layers,overwrite=True)
        elif save_mode == 2:
            self.model.save('./%d-cnn-population/winner_architecture.h5'%self.num_layers,overwrite=True)
        else:
            raise Exception('Illegal save_mode = %d'%save_mode)
    
    def train(self,dataset_train,dataset_valid):
        x_train, y_train = dataset_train
        x_valid, y_valid = dataset_valid
        self.model.fit(x_train,y_train,epochs=3,verbose=0)
        self.score = self.model.evaluate(x_valid,y_valid,verbose=0)[1]