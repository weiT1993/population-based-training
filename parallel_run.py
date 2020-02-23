import time
from worker import Worker
from mpi4py import MPI
from models import create_FCNN
from utils.datasets import get_dataset
import tensorflow as tf

def evolve(comm, worker, generation, epochs):
    population_size = comm.Get_size()
    score_before = worker.score
    worker.train(epochs=epochs)

    if worker.idx == population_size-1:
        print('-'*30,'Generation %d START'%generation,'-'*30,flush=True)
        for i in range(population_size-1):
            comm.send('Generation %d START'%generation, dest=i)
        print('Generation %d: Worker-%d changed from %.3f to %.3f'%(generation,worker.idx,score_before,worker.score),flush=True)

        max_score = worker.score
        max_idx = worker.idx
        max_model = worker.model
        for i in range(population_size-1):
            state = MPI.Status()
            worker_generation, worker_idx, worker_score = comm.recv(source=MPI.ANY_SOURCE,status=state)
            assert worker_generation == generation
            if worker_score>max_score:
                max_score = worker_score
                max_idx = worker_idx
        
        max_model.save('./checkpoints/generation_%d.h5'%generation)
        for i in range(population_size-1):
            comm.send((generation,max_idx,max_score), dest=i)
    else:
        state = MPI.Status()
        start_signal = comm.recv(source=population_size-1,status=state)
        assert start_signal == 'Generation %d START'%generation
        print('Generation %d: Worker-%d changed from %.3f to %.3f'%(generation,worker.idx,score_before,worker.score),flush=True)

        comm.send((generation,worker.idx,worker.score), dest=population_size-1)

        state = MPI.Status()
        master_generation, max_idx, max_score = comm.recv(source=population_size-1,status=state)
        assert master_generation == generation
    
    # print('Generation %d: Worker-%d START with %.3f'%(generation,worker.idx,worker.score),flush=True)
    
    if worker.idx != max_idx:
        max_model = tf.keras.models.load_model('./checkpoints/generation_%d.h5'%generation)
        worker.exploit(best_model=max_model)
        worker.explore()
        print('Generation %d: Worker-%d inherited from worker-%d = %.3f, exploited score = %.3f'%(
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
    dataset_train, dataset_valid, dataset_test = get_dataset(data_file='./data/power_15freq_7.3202.mat',time_range=300,concat=False)

    worker = Worker(idx=rank,model=model,dataset_train=dataset_train,dataset_valid=dataset_valid)

    for i in range(3):
        evolve(comm=comm, worker=worker, generation=i, epochs=3)