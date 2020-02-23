import time
from worker import Worker
from mpi4py import MPI

def evolve(comm, worker, generation, steps):
    population_size = comm.Get_size()

    if worker.idx == population_size-1:
        print('-'*30,'Generation %d START'%generation,'-'*30,flush=True)
        for i in range(population_size-1):
            comm.send('Generation %d START'%generation, dest=i)
        print('Generation %d: Worker-%d START with %.3f'%(generation,worker.idx,worker.score),flush=True)
        worker.train(steps=steps)

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
        print('Generation %d: Worker-%d START with %.3f'%(generation,worker.idx,worker.score),flush=True)
        worker.train(steps=steps)

        comm.send((generation,worker.idx,worker.score), dest=population_size-1)

        state = MPI.Status()
        master_generation, max_idx, max_score = comm.recv(source=population_size-1,status=state)
        assert master_generation == generation
    
    if worker.idx != max_idx:
        worker.exploit(max_score=max_score)
        worker.explore()
        print('Generation %d: Worker-%d inherited from worker-%d = %.3f, explore'%(generation,worker.idx,max_idx,max_score),flush=True)
    
    if worker.idx == population_size-1:
        for i in range(population_size-1):
            state = MPI.Status()
            end_signal = comm.recv(source=MPI.ANY_SOURCE,status=state)
            if end_signal != 'Generation %d DONE'%generation:
                raise Exception('Illegal end_signal : ',end_signal)
        print('-'*30,'Generation %d DONE'%generation,'-'*30,flush=True)
    else:
        comm.send('Generation %d DONE'%generation, dest=population_size-1)
        print('Generation %d: Worker-%d DONE with %.3f'%(generation,worker.idx,worker.score),flush=True)

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    worker = Worker(idx=rank)

    for i in range(3):
        evolve(comm=comm, worker=worker, generation=i, steps=20)