import argparse
import subprocess
import math
import random
from utils.helper_fun import read_file

def train(num_workers):
    child_processes = []
    for i in range(num_workers):
        p = subprocess.Popen(args=['python', 'worker.py','--idx','%d'%i,'--phase','train'])
        child_processes.append(p)
    
    for cp in child_processes:
        cp.wait()

def explore(num_workers,best_worker_ids):
    child_processes = []
    for i in range(num_workers):
        if i not in best_worker_ids:
            explore_target = random.choice(best_worker_ids)
            p = subprocess.Popen(args=['python', 'worker.py','--idx','%d'%i,'--phase','explore','--target-idx','%d'%explore_target])
            child_processes.append(p)
    
    for cp in child_processes:
        cp.wait()

def find_best_worker():
    scores_dict = read_file(filename='./population/scores.p')
    workers = list(scores_dict.keys())
    scores = list(scores_dict.values())
    scores, workers = zip(*sorted(zip(scores, workers),reverse=True))
    population_retain = math.ceil(0.2 * len(workers))
    best_workers = workers[:population_retain]
    best_scores = scores[:population_retain]
    return best_workers, best_scores

def evolve(generation,num_workers,max_generations):
    print('*'*50,'Generation %d START'%generation,'*'*50,flush=True)
    train(num_workers=num_workers)
    best_workers, best_scores = find_best_worker()
    print('Best workers : {}\nBest scores : {}'.format(best_workers,best_scores),flush=True)
    if generation<max_generations-1:
        subprocess.run(args=['rm','./population/scores.p'])
        explore(num_workers=num_workers,best_worker_ids=best_workers)

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