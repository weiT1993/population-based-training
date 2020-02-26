import time
import random
import argparse
import pickle
import os
import tensorflow as tf
from models import FC, CNN
from utils.helper_fun import read_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Worker instance')
    parser.add_argument('--idx', metavar='N', type=int,help='Worker Index')
    parser.add_argument('--phase', type=str, choices=['init','train','explore','conclude','won'])
    parser.add_argument('--target-idx', metavar='N', default=-1, type=int,help='Best Worker Index')
    parser.add_argument('--num-layers', metavar='N', type=int,help='Number of NN layers')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if args.phase == 'init':
        print('Worker-%d instantiated'%args.idx,flush=True)
        worker = FC(worker_idx=args.idx,num_layers=args.num_layers)
        worker.random_model()
        worker.save_model(save_mode=0)
    elif args.phase == 'train':
        data_dict = read_file(filename='./data/dataset.p')
        worker = FC(worker_idx=args.idx,num_layers=args.num_layers)
        worker.load_model(dataset_valid=data_dict['valid'])
        old_score = worker.score
        worker.train(dataset_train=data_dict['train'],dataset_valid=data_dict['valid'])
        worker.save_model(save_mode=0)
        new_score = worker.score
        print('Worker-%d score trained from %.5f --> %.5f'%(args.idx,old_score,new_score),flush=True)
        # NOTE: avoid racing
        time.sleep(random.random())
        pickle.dump({args.idx:new_score},open('./%d-fc-population/scores.p'%args.num_layers,'ab'))
    elif args.phase == 'explore':
        data_dict = read_file(filename='./data/dataset.p')
        good_worker = FC(worker_idx=args.target_idx,num_layers=args.num_layers)
        good_worker.load_model(dataset_valid=data_dict['valid'])
        worker = FC(worker_idx=args.idx,num_layers=args.num_layers)
        worker.load_model(dataset_valid=data_dict['valid'])
        old_score = worker.score
        worker.explore_model(good_model=good_worker.model,dataset_valid=data_dict['valid'])
        worker.save_model(save_mode=0)
        print('Worker-%d (%.5f) explore worker-%d (%.5f), initial score = %.5f'%(worker.worker_idx,old_score,
        good_worker.worker_idx,good_worker.score,worker.score),flush=True)
    elif args.phase == 'conclude':
        data_dict = read_file(filename='./data/dataset.p')
        worker = FC(worker_idx=args.idx,num_layers=args.num_layers)
        worker.load_model(dataset_valid=data_dict['valid'])
        worker.save_model(save_mode=1)
        print('Worker-%d has the best architecture. Score : %.5f'%(args.idx,worker.score),flush=True)
    elif args.phase == 'won':
        data_dict = read_file(filename='./data/dataset.p')
        worker = FC(worker_idx=args.idx,num_layers=args.num_layers)
        worker.load_model(dataset_valid=data_dict['valid'])
        worker.save_model(save_mode=2)
        print('Worker-%d has won! Score : %.5f'%(args.idx,worker.score),flush=True)
    else:
        raise Exception('Illegal phase = %s'%args.phase)