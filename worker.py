import tensorflow as tf
from models import explore_FCNN, create_FCNN

class Worker():
    def __init__(self, idx, model, dataset_train, dataset_valid):
        self.idx = idx
        self.model = model
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        self.score = -1

    def train(self, epochs):
        x_train, y_train = self.dataset_train
        self.model.fit(x_train, y_train, epochs=epochs, verbose=0)
        x_valid, y_valid = self.dataset_valid
        self.score = self.model.evaluate(x_valid, y_valid, verbose=0)[1]
        self.winner_idx = self.idx

    def exploit(self,best_model_h5):
        best_model = tf.keras.models.load_model(best_model_h5, custom_objects={'leaky_relu': tf.nn.leaky_relu,
        'crelu_v2':tf.nn.crelu})
        self.best_model = best_model
    
    def explore(self):
        old_score = self.score
        x_valid, y_valid = self.dataset_valid
        best_score = self.best_model.evaluate(x_valid, y_valid, verbose=0)[1]
        if best_score < 0.8:
            self.model = create_FCNN(num_layers=3,worker_idx=self.idx)
            print('Worker-%d resample, %.5f'%(self.idx,old_score),flush=True)
        else:
            self.model = explore_FCNN(good_model=self.best_model, bad_model=self.model, worker_idx=self.idx)
            new_score = self.model.evaluate(x_valid, y_valid, verbose=0)[1]
            print('Worker-%d explore, %.5f-->%.5f'%(self.idx,old_score,new_score),flush=True)