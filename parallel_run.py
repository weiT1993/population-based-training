import time
import random
import pickle
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
        for i in range(population_size-1):
            if i == max_idx:
                comm.send('Generation %d WINNER'%generation, dest=i)
            else:
                comm.send('Generation %d LOSER'%generation, dest=i)
    else:
        comm.send((generation,worker.idx,worker.score), dest=population_size-1)
        state = MPI.Status()
        comparison_result = comm.recv(source=population_size-1,status=state)
        if comparison_result == 'Generation %d WINNER'%generation:
            worker.model.save('./checkpoints/generation_%d.h5'%generation,overwrite=True)
            print('Generation %d: Worker-%d saved best model'%(generation,worker.idx),flush=True)
            worker.model.summary()
        elif comparison_result == 'Generation %d LOSER'%generation:
            pass
        else:
            raise Exception('Illegal comparison_result received :',comparison_result)

def evolve(comm, worker, generation, epochs):
    train(comm, worker, generation, epochs)
    save_winner(comm, worker, generation)

def _evolve(comm, worker, generation, epochs):
    population_size = comm.Get_size()
    score_before = worker.score
    worker.train(epochs=epochs)

    if worker.idx == population_size-1:
        print('-'*30,'Generation %d START'%generation,'-'*30,flush=True)
        for i in range(population_size-1):
            comm.send('Generation %d START'%generation, dest=i)
        print('Generation %d: Worker-%d changed from %.5f to %.5f'%(generation,worker.idx,score_before,worker.score),flush=True)

        max_score = worker.score
        max_idx = worker.idx
        for i in range(population_size-1):
            state = MPI.Status()
            worker_generation, worker_idx, worker_score = comm.recv(source=MPI.ANY_SOURCE,status=state)
            assert worker_generation == generation
            if worker_score>max_score:
                max_score = worker_score
                max_idx = worker_idx
        
        for i in range(population_size-1):
            comm.send((generation,max_idx,max_score), dest=i)
    else:
        state = MPI.Status()
        start_signal = comm.recv(source=population_size-1,status=state)
        assert start_signal == 'Generation %d START'%generation
        print('Generation %d: Worker-%d changed from %.5f to %.5f'%(generation,worker.idx,score_before,worker.score),flush=True)

        comm.send((generation,worker.idx,worker.score), dest=population_size-1)

        state = MPI.Status()
        master_generation, max_idx, max_score = comm.recv(source=population_size-1,status=state)
        assert master_generation == generation
    
    # print('Generation %d: Worker-%d START with %.3f'%(generation,worker.idx,worker.score),flush=True)
    
    if worker.idx != max_idx:
        worker.exploit(best_model_h5='./checkpoints/generation_%d.h5'%generation)
        worker.explore()
        print('Generation %d: Worker-%d inherited from worker-%d = %.5f, exploited score = %.5f'%(
            generation,worker.idx,max_idx,max_score,worker.score),flush=True)
    
    if worker.idx == population_size-1:
        for i in range(population_size-1):
            state = MPI.Status()
            end_signal = comm.recv(source=MPI.ANY_SOURCE,status=state)
            if end_signal != 'Generation %d DONE'%generation:
                raise Exception('Illegal end_signal : ',end_signal)
        # print('Generation %d: Worker-%d DONE with %.3f'%(generation,worker.idx,worker.score),flush=True)
        print('-'*30,'Generation %d DONE'%generation,'-'*30,flush=True)
    else:
        comm.send('Generation %d DONE'%generation, dest=population_size-1)
        # print('Generation %d: Worker-%d DONE with %.3f'%(generation,worker.idx,worker.score),flush=True)

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    model = create_FCNN(num_layers=3)
    data_dict = read_file(filename='./data/dataset.p')
    dataset_train = data_dict['train']
    dataset_valid = data_dict['valid']
    dataset_test = data_dict['test']

    # print('%d train, %d val, %d test'%(len(dataset_train[0]),len(dataset_valid[0]),len(dataset_test[0])))

    worker = Worker(idx=rank,model=model,dataset_train=dataset_train,dataset_valid=dataset_valid)

    for i in range(3):
        evolve(comm=comm, worker=worker, generation=i, epochs=1)