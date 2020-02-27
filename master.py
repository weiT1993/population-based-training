import argparse
import subprocess
import math
import random
import pickle
from utils.helper_fun import read_file
import os
from time import time

def train(num_workers,num_layers,model):
    child_processes = []
    for i in range(num_workers):
        p = subprocess.Popen(args=['python', 'worker.py',
        '--model',model,
        '--idx','%d'%i,'--target-idx','%d'%-1,
        '--phase','train',
        '--num-layers','%d'%num_layers])
        child_processes.append(p)
    
    for cp in child_processes:
        cp.wait()

def explore(num_workers,model,num_layers,best_worker_ids,worst_worker_ids):
    child_processes = []
    for i in range(num_workers):
        if i in worst_worker_ids:
            explore_target = random.choice(best_worker_ids)
            p = subprocess.Popen(args=['python', 'worker.py',
            '--model',model,
            '--idx','%d'%i,'--target-idx','%d'%explore_target,
            '--phase','explore',
            '--num-layers','%d'%num_layers])
            child_processes.append(p)
    
    for cp in child_processes:
        cp.wait()

def compete(num_layers,model):
    scores_dict = read_file(filename='./%d-%s-population/scores.p'%(num_layers,model))
    workers = list(scores_dict.keys())
    scores = list(scores_dict.values())
    scores, workers = zip(*sorted(zip(scores, workers),reverse=True))
    population_retain = math.ceil(0.1 * len(workers))
    population_dump = math.ceil(0.5 * len(workers))
    best_workers = workers[:population_retain]
    best_scores = scores[:population_retain]
    worst_workers = workers[-population_dump:]
    worst_scores = scores[-population_dump:]
    
    leaderboard = read_file(filename='./%d-%s-population/leaderboard.p'%(num_layers,model))
    for i in workers:
        if i in best_workers and i in leaderboard:
            leaderboard[i] += 1
        elif i in best_workers and i not in leaderboard:
            leaderboard[i] = 1
        else:
            leaderboard[i] = 0
    pickle.dump(leaderboard,open('./%d-%s-population/leaderboard.p'%(num_layers,model),'ab'))
    print('Leaderboard:',leaderboard)
    winner_idx = -1
    for worker_idx in leaderboard:
        if leaderboard[worker_idx] >= 10:
            winner_idx = worker_idx
    return best_workers, best_scores, worst_workers, worst_scores, winner_idx

def evolve(generation,model,num_workers,num_layers,max_generations):
    print('*'*50,'Generation %d START'%generation,'*'*50,flush=True)
    
    train(num_workers=num_workers,num_layers=num_layers,model=model)
    
    best_workers, best_scores, worst_workers, worst_scores, winner_idx = compete(num_layers=num_layers,model=model)
    print('Best workers : {}\nBest scores : {}\nWorst workers : {}\nWorst scores : {}'.format(best_workers,best_scores,
    worst_workers,worst_scores),flush=True)
    
    if generation<max_generations-1 and winner_idx==-1:
        subprocess.run(args=['rm','./%d-%s-population/scores.p'%(num_layers,model)])
        explore(num_workers=num_workers,model=model,num_layers=num_layers,best_worker_ids=best_workers,worst_worker_ids=worst_workers)
        return best_workers, winner_idx
    else:
        return best_workers, winner_idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Worker instance')
    parser.add_argument('--model', type=str, choices=['cnn','fc'])
    parser.add_argument('--num-workers', metavar='N', type=int,help='Number of parallel workers')
    parser.add_argument('--num-layers', metavar='N', type=int,help='Number of NN layers')
    parser.add_argument('--max-generations', metavar='N', type=int,help='Number of generations')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if not os.path.exists('./%d-%s-population'%(args.num_layers,args.model)):
        os.makedirs('./%d-%s-population'%(args.num_layers,args.model))

    init_processes = []
    for i in range(args.num_workers):
        p = subprocess.Popen(args=['python', 'worker.py',
        '--model',args.model,
        '--idx','%d'%i,'--target-idx','%d'%-1,
        '--phase','init',
        '--num-layers','%d'%int(args.num_layers)])
        init_processes.append(p)
    for p in init_processes:
        p.wait()

    begin = time()
    for i in range(args.max_generations):
        best_workers, winner_idx = evolve(generation=i,model=args.model,num_workers=args.num_workers,num_layers=args.num_layers,max_generations=args.max_generations)
        time_elapsed = time() - begin
        eta = time_elapsed/(i+1)*args.max_generations - time_elapsed
        print('ETA = %d seconds'%eta,flush=True)
        if winner_idx != -1:
            subprocess.run(args=['python', 'worker.py',
            '--model',args.model,
            '--idx','%d'%winner_idx,'--target-idx','%d'%-1,
            '--phase','won',
            '--num-layers','%d'%int(args.num_layers)])
            break
    
    if winner_idx==-1:
        subprocess.run(args=['python', 'worker.py',
        '--model',args.model,
        '--idx','%d'%best_workers[0],'--target-idx','%d'%-1,
        '--phase','conclude',
        '--num-layers','%d'%int(args.num_layers)])
