import time
from worker import Worker
from mpi4py import MPI

def evolve(comm, worker, generation, steps):
    population_size = comm.Get_size()
    worker.train(steps=steps)

    if worker.idx == population_size-1:
        max_score = worker.score
        max_idx = worker.idx
        for i in range(population_size-1):
            state = MPI.Status()
            worker_idx, worker_score = comm.recv(source=MPI.ANY_SOURCE,status=state)
            if worker_score>max_score:
                max_score = worker_score
                max_idx = worker_idx
        for i in range(population_size-1):
            comm.send((max_idx,max_score), dest=i)
    else:
        comm.send((worker.idx,worker.score), dest=population_size-1)
        state = MPI.Status()
        max_idx, max_score = comm.recv(source=population_size-1,status=state)
    
    if worker.idx != max_idx:
        worker.exploit(max_score=max_score)
        worker.explore()
        print('Generation %d: Worker-%d inherited from worker-%d = %.3f, explore'%(generation,worker.idx,max_idx,max_score),flush=True)

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    worker = Worker(idx=rank)

    for i in range(3):
        if worker.idx == size-1:
            print('-'*30,'Generation %d begins'%i,'-'*30,flush=True)
        
        evolve(comm=comm, worker=worker, generation=i, steps=20)
        
        if worker.idx == size-1:
            print('-'*30,'Generation %d DONE'%i,'-'*30,flush=True)