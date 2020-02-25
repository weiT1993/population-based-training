import time
import random
import argparse
import pickle
import tensorflow as tf
from models import FC
from utils.helper_fun import read_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Worker instance')
    parser.add_argument('--idx', metavar='N', type=int,help='Worker Index')
    parser.add_argument('--phase', type=str, choices=['init','train','explore','exploit'])
    parser.add_argument('--best-idx', metavar='N', default=-1, type=int,help='Best Worker Index')
    args = parser.parse_args()

    if args.phase == 'init':
        print('Worker-%d instantiated'%args.idx,flush=True)
        worker = FC(worker_idx=args.idx)
        worker.random_model(num_layers=3)
        worker.save_model()
    elif args.phase == 'train':
        worker = FC(worker_idx=args.idx)
        data_dict = read_file(filename='./data/dataset.p')
        worker.load_model(model_h5='./population/worker_%d.h5'%args.idx,dataset_valid=data_dict['valid'])
        old_score = worker.score
        worker.train(dataset_train=data_dict['train'],dataset_valid=data_dict['valid'])
        worker.save_model()
        new_score = worker.score
        print('Worker-%d score trained from %.5f --> %.5f'%(args.idx,old_score,new_score),flush=True)
        # NOTE: avoid racing
        time.sleep(random.random())
        pickle.dump({args.idx:new_score},open('./population/scores.p','ab'))
    elif args.phase == 'explore':
        best_worker = FC(worker_idx=args.idx)
        data_dict = read_file(filename='./data/dataset.p')
        best_worker.load_model(model_h5='./population/worker_%d.h5'%args.best_idx,dataset_valid=data_dict['valid'])
        best_score = best_worker.score
        print('Worker-%d explore best_score = %.5f'%(args.idx,best_score),flush=True)
    else:
        print('Worker-%d exploit'%args.idx,flush=True)