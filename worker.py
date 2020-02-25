import time
import argparse
import tensorflow as tf
from models import FC

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Worker instance')
    parser.add_argument('--idx', metavar='N', type=int,help='Worker Index')
    parser.add_argument('--phase', type=str, choices=['init','train','explore','exploit'])
    args = parser.parse_args()

    if args.phase == 'init':
        print('Worker %d instantiated'%args.idx,flush=True)
        worker = FC(architecture={'num_layers':3})
    elif args.phase == 'train':
        print('Worker %d train'%args.idx,flush=True)
    elif args.phase == 'explore':
        print('Worker %d explore'%args.idx,flush=True)
    else:
        print('Worker %d exploit'%args.idx,flush=True)