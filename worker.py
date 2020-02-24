import tensorflow as tf

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
        best_model = tf.keras.models.load_model(best_model_h5)
        old_score = self.score
        print('Worker-%d exploit, old score = %.5f'%(self.idx,old_score))
        x_valid, y_valid = self.dataset_valid
        new_score = best_model.evaluate(x_valid, y_valid, verbose=0)[1]
        print('Worker-%d exploit, new score = %.5f'%(self.idx,new_score))
        print('Worker-%d exploit, %.5f-->%.5f'%(self.idx,old_score,new_score),flush=True)
    
    def explore(self):
        # # self.model.summary()
        # # self.best_model.summary()
        # best_weights = self.best_model.weights
        # print(type(best_weights))
        # print(type(best_weights[0]))
        # print(best_weights[0])
        1+1