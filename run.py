import logging
import numpy as np
import time
import operator
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager

from worker import Worker

# unfortunately multiprocessing module can't unpickle lambda functions
def obj(theta):
    return 1.2 - np.sum(theta**2)
    
def surrogate_obj(theta, h):
    return 1.2 - np.sum(h*theta**2)

def run(worker, steps, theta_dict, Q_dict, loss_dict):
    """start worker object asychronously"""
    for step in range(steps):
        worker.step(vanilla=True,use_loss=False) # one step of SGD
        worker.eval() # evaluate current model
        
        if step % 10 == 0:
            do_explore = worker.exploit()                
            if do_explore:
                worker.explore()
                                        
        worker.update()
    
    # TODO: there should be better ways
    time.sleep(worker.idx) # to avoid race conditions
    
    _theta_dict = theta_dict[0]
    _Q_dict = Q_dict[0]
    _loss_dict = loss_dict[0]
    _theta_dict[worker.idx] = worker.theta_history
    _Q_dict[worker.idx] = worker.Q_history
    _loss_dict[worker.idx] = worker.loss_history
    theta_dict[0] = _theta_dict
    Q_dict[0] = _Q_dict
    loss_dict[0] = _loss_dict

def plot(title, type, history, steps, population_sizes):
    
    for population_size in population_sizes:
        if type == 'Q':
            plt.plot(history[population_size], lw=0.7, label=str(population_size))
        else:
            plt.scatter(np.arange(0,steps+1), history[population_size], label=str(population_size), s=2)
    
    if type == 'Q':
        plt.axhline(y=1.2, linestyle='dotted', color='k')
        
    axes = plt.gca()
    axes.set_xlim([0,steps])
    if type == 'Q':
        axes.set_ylim([0.0, 1.21])
    
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel(type)
    plt.legend(loc='upper right')
    
    plt.show()
    
def plot_theta(title, history, steps, population_sizes):
    
    for population_size in population_sizes:
        x = [_[0] for _ in history[population_size]]
        y = [_[1] for _ in history[population_size]]
        plt.scatter(x, y, s=2, label=str(population_size))
        
    plt.title(title)
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.legend(loc='upper right')
    
    plt.show()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d %(name)s %(message)s',
                        datefmt="%M:%S")
    
    Q_dict_with_size = {} # stores {population_size: Q_dict}
    theta_dict_with_size = {} # stores {population_size: theta}
    loss_dict_with_size = {} 
    
    population_sizes = [2]

    for population_size in population_sizes:

        pop_score = Manager().list() # create a proxy for shared objects between processes
        pop_score.append({})
        
        pop_params = Manager().list()
        pop_params.append({})
        
        steps = 100

        Population = [
                    Worker(
                        idx=i, 
                        obj=obj, 
                        surrogate_obj=surrogate_obj, 
                        h=np.random.rand(2), 
                        theta=np.random.rand(2), 
                        pop_score=pop_score, 
                        pop_params=pop_params,
                        use_logger=False, # unfortunately difficult to use logger in multiprocessing
                        log_frequency=50,
                        asynchronous=True, # enable shared memory between spawned processes
                        )
                        for i in range(population_size)
                        ]
        
        theta_dict = Manager().list()
        theta_dict.append({})
        loss_dict = Manager().list()
        loss_dict.append({})
        Q_dict = Manager().list()
        Q_dict.append({})

        processes = []
        # create the processes to run asynchronously
        for worker in Population:
            _p = Process(
                    target=run, 
                    args=(worker,steps,theta_dict,Q_dict,loss_dict)
                    )
            processes.append(_p)
        
        # start the processes
        for i in range(population_size):
            processes[i].start()
        for i in range(population_size): # join to prevent Manager to shutdown
            processes[i].join()
        
        # find agent with best performance
        best_worker_idx = max(pop_score[0].items(), key=operator.itemgetter(1))[0]

        # save best agent/worker for a given population size
        Q_dict_with_size[population_size] = Q_dict[0][best_worker_idx]
        theta_dict_with_size[population_size] = theta_dict[0][best_worker_idx]
        loss_dict_with_size[population_size] = loss_dict[0][best_worker_idx]

    # plot('Q per step for various population sizes', 'Q', Q_dict_with_size, steps, population_sizes)
    # plot_theta('theta per step for various population sizes', theta_dict_with_size, steps, population_sizes)
    # plot('loss per step for various population sizes', 'loss', loss_dict_with_size, steps, population_sizes)