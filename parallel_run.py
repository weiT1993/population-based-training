import time
import random
import pickle
import re
import os
from worker import Worker
from mpi4py import MPI
from models import create_FCNN
from utils.datasets import get_dataset
from utils.helper_fun import read_file
import tensorflow as tf

def train(comm, worker, generation, epochs):
    old_score = worker.score
    worker.train(epochs=epochs)
    new_score = worker.score
    population_size = comm.Get_size()
    if worker.idx == population_size-1:
        print('-'*30,'Generation %d START'%generation,'-'*30,flush=True)
        for i in range(population_size-1):
            comm.send('Generation %d START'%generation, dest=i)
    else:
        state = MPI.Status()
        start_signal = comm.recv(source=population_size-1,status=state)
        assert start_signal == 'Generation %d START'%generation
    print('Generation %d: Worker-%d changed from %.5f to %.5f'%(generation,worker.idx,old_score,new_score),flush=True)

def save_winner(comm, worker, generation):
    population_size = comm.Get_size()
    if worker.idx == population_size-1:
        max_score = worker.score
        max_idx = worker.idx
        for i in range(population_size-1):
            state = MPI.Status()
            worker_generation, worker_idx, worker_score = comm.recv(source=MPI.ANY_SOURCE,status=state)
            assert worker_generation == generation
            if worker_score>max_score:
                max_score = worker_score
                max_idx = worker_idx
        print('Generation %d: Worker-%d has best model, score = %.5f'%(generation,max_idx,max_score),flush=True)
        if max_idx == worker.idx:
            worker.model.save('./checkpoints/generation_%d.h5'%generation,overwrite=True)
            print('Generation %d: Worker-%d saved best model'%(generation,worker.idx),flush=True)
        worker.winner_idx = (generation, max_idx)
        for i in range(population_size-1):
            comm.send('Generation %d WINNER: %d'%(generation, max_idx), dest=i)
    else:
        comm.send((generation,worker.idx,worker.score), dest=population_size-1)
        state = MPI.Status()
        comparison_result = comm.recv(source=population_size-1,status=state)
        pattern = re.compile(r'Generation ([0-9]+) WINNER: ([0-9]+)')
        m = pattern.match(comparison_result)
        if m == None:
            raise Exception('Illegal comparison_result:',comparison_result)
        else:
            comparison_generation = int(m.group(1))
            max_idx = int(m.group(2))
            assert comparison_generation == generation
            if max_idx == worker.idx:
                worker.model.save('./checkpoints/generation_%d.h5'%generation,overwrite=True)
                print('Generation %d: Worker-%d saved best model'%(generation,worker.idx),flush=True)
            worker.winner_idx = (generation, max_idx)

def exploit_winner(comm, worker, generation):
    # print('Generation {:d}: Worker-{:d} has winner_idx = {}'.format(generation,worker.idx,worker.winner_idx),flush=True)
    # NOTE: Should be better ways
    random.seed(worker.idx)
    assert worker.winner_idx[0] == generation
    if worker.winner_idx[1] == worker.idx:
        pass
    else:
        while not os.path.isfile('./checkpoints/generation_%d.h5'%generation):
            time.sleep(1)
        time.sleep(random.random())
        worker.exploit(best_model_h5='./checkpoints/generation_%d.h5'%generation)
        worker.explore()

def evolve(comm, worker, generation, epochs):
    train(comm, worker, generation, epochs)
    save_winner(comm, worker, generation)
    exploit_winner(comm, worker, generation)

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == size-1:
        data_dict = read_file(filename='./data/dataset.p')
        for i in range(size-1):
            comm.send(data_dict, dest=i)
    else:
        state = MPI.Status()
        data_dict = comm.recv(source=size-1,status=state)
    
    dataset_train = data_dict['train']
    dataset_valid = data_dict['valid']
    dataset_test = data_dict['test']

    model = create_FCNN(num_layers=3, worker_idx=rank)
    worker = Worker(idx=rank,model=model,dataset_train=dataset_train,dataset_valid=dataset_valid)

    for i in range(5):
        evolve(comm=comm, worker=worker, generation=i, epochs=3)