import argparse
import subprocess
import math
import random
import pickle
from utils.helper_fun import read_file

def train(num_workers):
    child_processes = []
    for i in range(num_workers):
        p = subprocess.Popen(args=['python', 'worker.py','--idx','%d'%i,'--phase','train'])
        child_processes.append(p)
    
    for cp in child_processes:
        cp.wait()

def explore(num_workers,best_worker_ids,worst_worker_ids):
    child_processes = []
    for i in range(num_workers):
        if i in worst_worker_ids:
            explore_target = random.choice(best_worker_ids)
            p = subprocess.Popen(args=['python', 'worker.py','--idx','%d'%i,'--phase','explore','--target-idx','%d'%explore_target])
            child_processes.append(p)
    
    for cp in child_processes:
        cp.wait()

def compete():
    scores_dict = read_file(filename='./population/scores.p')
    workers = list(scores_dict.keys())
    scores = list(scores_dict.values())
    scores, workers = zip(*sorted(zip(scores, workers),reverse=True))
    population_retain = math.ceil(0.1 * len(workers))
    population_dump = math.ceil(0.5 * len(workers))
    best_workers = workers[:population_retain]
    best_scores = scores[:population_retain]
    worst_workers = workers[-population_dump:]
    worst_scores = scores[-population_dump:]
    
    leaderboard = read_file(filename='./population/leaderboard.p')
    for i in workers:
        if i in best_workers and i in leaderboard:
            leaderboard[i] += 1
        elif i in best_workers and i not in leaderboard:
            leaderboard[i] = 1
        else:
            leaderboard[i] = 0
    pickle.dump(leaderboard,open('./population/leaderboard.p','ab'))
    print('Leaderboard:',leaderboard)
    winner_idx = -1
    for worker_idx in leaderboard:
        if leaderboard[worker_idx] >= 10:
            winner_idx = worker_idx
    return best_workers, best_scores, worst_workers, worst_scores, winner_idx

def evolve(generation,num_workers,max_generations):
    print('*'*50,'Generation %d START'%generation,'*'*50,flush=True)
    train(num_workers=num_workers)
    best_workers, best_scores, worst_workers, worst_scores, winner_idx = compete()
    print('Best workers : {}\nBest scores : {}\nWorst workers : {}\nWorst scores : {}'.format(best_workers,best_scores,
    worst_workers,worst_scores),flush=True)
    if generation<max_generations-1 and winner_idx==-1:
        subprocess.run(args=['rm','./population/scores.p'])
        explore(num_workers=num_workers,best_worker_ids=best_workers,worst_worker_ids=worst_workers)
        return best_workers, winner_idx
    else:
        return best_workers, winner_idx

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
        best_workers, winner_idx = evolve(generation=i,num_workers=args.num_workers,max_generations=args.num_generations)
        if winner_idx != -1:
            subprocess.run(args=['python', 'worker.py','--idx','%d'%winner_idx,'--phase','won'])
            break
    
    if winner_idx==-1:
        subprocess.run(args=['python', 'worker.py','--idx','%d'%best_workers[0],'--phase','conclude'])