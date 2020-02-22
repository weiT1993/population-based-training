import logging
import numpy as np
import operator

class Worker():
    def __init__(self, idx, obj, surrogate_obj, h, theta, pop_score, pop_params, use_logger, log_frequency, asynchronous):
        self.idx = idx
        self.log_frequency = log_frequency

        self.use_logger = use_logger
        if use_logger:
            self.logger = logging.getLogger("Worker-{}".format(self.idx))
        else:
            print("Beginning Worker-{}".format(self.idx))
        
        self.asynchronous = asynchronous

        self.obj = obj
        self.surrogate_obj = surrogate_obj
        self.theta = theta
        self.h = h
        
        self.score = 0 # current score
        self.loss = 0
        
        self.pop_score = pop_score # reference to population statistics
        self.pop_params = pop_params
        
        # for plotting
        self.theta_history = []
        self.Q_history = []
        self.loss_history = []
        
        self.rms = 0 # for rmsprop
        
        self.update() # intialize population

    def update(self):
        """update worker stats in global dictionary"""
        if not self.asynchronous:
            self.pop_score[self.idx] = self.score
            self.pop_params[self.idx] = (np.copy(self.theta), np.copy(self.h)) # np arrays are mutable
        else:
            self.proxy_sync(pull=False,push=True)
            
        self.theta_history.append(np.copy(self.theta))
        self.Q_history.append(self.score)
        self.loss_history.append(self.loss)
        
        if len(self.Q_history) % self.log_frequency == 0:
            if self.use_logger:
                self.logger.info("Q = {:0.2f} ({:0.2f}%)".format(self.score, self.score * 100 / 1.2))
            else:
                print("Worker-{} Step {} Q = {:0.2f} ({:0.2f}%)".format(
                                                            self.idx, 
                                                            len(self.Q_history),
                                                            self.score, 
                                                            self.score * 100 / 1.2),
                                                            )
    
    def proxy_sync(self, pull, push):
        """for asynchronous workers, we need to sync the values to the shared proxies
        https://docs.python.org/2/library/multiprocessing.html#multiprocessing.managers.SyncManager.list
        """
        
        if pull: # grab newest copy of pop_params
            return self.pop_score[0], self.pop_params[0]

        if push: # update newest copy
            _pop_score = self.pop_score[0]
            _pop_params = self.pop_params[0]
            
            _pop_score[self.idx] = self.score
            _pop_params[self.idx] = (np.copy(self.theta), np.copy(self.h))
            
            self.pop_score[0] = _pop_score
            self.pop_params[0] = _pop_params

    def step(self, vanilla, use_loss):
        """one step of SGD"""
        # TODO: Need to change this to one step of NN training
        decay_rate = 0.9
        alpha = 0.01
        eps = 1e-5
        
        d_surrogate_obj = -2.0 * self.h * self.theta
        
        if use_loss:
            self.loss = (self.obj(self.theta)-self.surrogate_obj(self.theta, self.h))**2
            d_loss = 2 * (self.obj(self.theta)-self.surrogate_obj(self.theta, self.h)) * d_surrogate_obj
        else: # paper maximized Q (did not use loss function)
            d_loss = -d_surrogate_obj # negative for gradient ascent
        
        if vanilla:
            self.theta -= d_loss * alpha
        else:
            self.rms = decay_rate * self.rms + (1-decay_rate) * d_loss**2
            self.theta -= alpha * d_loss / (np.sqrt(self.rms) + eps)
    
    def eval(self):
        """metric we want to optimize e.g mean episodic return or validation set performance"""
        self.score = self.obj(self.theta)
        return self.score
    
    def exploit(self):
        """copy weights, hyperparams from the member in the population with the highest performance"""
        if self.asynchronous:
            pop_score, pop_params = self.proxy_sync(pull=True,push=False)
        else:
            pop_score = self.pop_score
            pop_params = self.pop_params
            
        best_worker_idx = max(pop_score.items(), key=operator.itemgetter(1))[0]
        print(type(pop_score),pop_score)
        # print(operator.itemgetter(1))
        # print(max(pop_score.items(), key=operator.itemgetter(1)))
        # print(best_worker_idx)
        if best_worker_idx != self.idx:
            # print(self.idx, pop_score) enable to check if shared memory is being updated
            
            best_worker_theta, best_worker_h = pop_params[best_worker_idx]
            self.theta = np.copy(best_worker_theta)
            
            if self.use_logger:
                self.logger.info("Inherited optimal weights from Worker-{}".format(best_worker_idx))
            else:
                print("Worker-{} Inherited optimal weights from Worker-{}".format(self.idx, best_worker_idx))
            return True
        return False
    
    def explore(self):
        # NOTE: assuming hyperparameters centered around a normal distribution tend to perform similarly
        """perturb hyperparameters with noise from a normal distribution"""
        eps = np.random.randn(*self.h.shape) * 0.1
        self.h += eps