import argparse
import subprocess
from utils.helper_fun import read_file

def train(num_workers):
    child_processes = []
    for i in range(num_workers):
        p = subprocess.Popen(args=['python', 'worker.py','--idx','%d'%i,'--phase','train'])
        child_processes.append(p)
    
    for cp in child_processes:
        cp.wait()

def explore(num_workers,best_worker_idx):
    child_processes = []
    for i in range(num_workers):
        if i != best_worker_idx:
            p = subprocess.Popen(args=['python', 'worker.py','--idx','%d'%i,'--phase','explore','--best-idx','%d'%best_worker_idx])
            child_processes.append(p)
    
    for cp in child_processes:
        cp.wait()

def exploit(num_workers):
    child_processes = []
    for i in range(num_workers):
        p = subprocess.Popen(args=['python', 'worker.py','--idx','%d'%i,'--phase','exploit'])
        child_processes.append(p)
    
    for cp in child_processes:
        cp.wait()

def find_best_worker():
    scores = read_file(filename='./population/scores.p')
    best_score = 0
    best_worker = -1
    for worker_idx in scores:
        if scores[worker_idx]>best_score:
            best_score = scores[worker_idx]
            best_worker = worker_idx
    subprocess.run(args=['rm','./population/scores.p'])
    return best_worker, best_score

def evolve(generation,num_workers,max_generations):
    print('*'*50,'Generation %d START'%generation,'*'*50,flush=True)
    train(num_workers=num_workers)
    if generation<max_generations-1:
        best_worker, best_score = find_best_worker()
        print('Worker-%d has the best score = %.5f'%(best_worker,best_score))
        explore(num_workers=num_workers,best_worker_idx=best_worker)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Worker instance')
    parser.add_argument('--num-workers', metavar='N', type=int,help='Number of parallel workers')
    parser.add_argument('--num-generations', metavar='N', type=int,help='Number of generations')
    args = parser.parse_args()

    init_processes = []
    for i in range(args.num_workers):
        p = subprocess.Popen(args=['python', 'worker.py','--idx','%d'%i,'--phase','init'])
        init_processes.append(p)
    for p in init_processes:
        p.wait()

    for i in range(args.num_generations):
        evolve(generation=i,num_workers=args.num_workers,max_generations=args.num_generations)