import logging
import numpy as np
import operator
import random
import time

class Worker():
    def __init__(self, idx):
        self.idx = idx
        self.score = 0

    def train(self, steps):
        for step in range(steps):
            self.score += random.random()

    def exploit(self,max_score):
        self.score = max_score
    
    def explore(self):
        1+1