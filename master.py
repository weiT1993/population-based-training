import argparse
import subprocess

def train(num_workers):
    child_processes = []
    for i in range(num_workers):
        p = subprocess.Popen(args=['python', 'worker.py','--idx','%d'%i,'--phase','train'])
        child_processes.append(p)
    
    for cp in child_processes:
        cp.wait()

def explore(num_workers):
    child_processes = []
    for i in range(num_workers):
        p = subprocess.Popen(args=['python', 'worker.py','--idx','%d'%i,'--phase','explore'])
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

def evolve(generation,num_workers,max_generations):
    print('*'*50,'Generation %d START'%generation,'*'*50,flush=True)
    train(num_workers=num_workers)
    if generation<max_generations-1:
        explore(num_workers=num_workers)
        exploit(num_workers=num_workers)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WOrker instance')
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