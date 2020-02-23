import random

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

    def exploit(self,max_score):
        self.score = max_score
    
    def explore(self):
        1+1