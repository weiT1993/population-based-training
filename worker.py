import time
import random
import argparse
import pickle
import os
import tensorflow as tf
from models import Model
from utils.helper_fun import read_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Worker instance')
    parser.add_argument('--model', type=str, choices=['cnn','fc'])
    parser.add_argument('--idx', metavar='N', type=int,help='Worker Index')
    parser.add_argument('--target-idx', metavar='N', type=int,help='Best Worker Index')
    parser.add_argument('--phase', type=str, choices=['init','train','explore','conclude','won'])
    parser.add_argument('--num-layers', metavar='N', type=int,help='Number of NN layers')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if args.phase == 'init':
        print('Worker-%d instantiated'%args.idx,flush=True)
        worker = Model(worker_idx=args.idx,num_layers=args.num_layers,model=args.model)
        worker.random_model()
        worker.save_model(save_mode=0)
    elif args.phase == 'train':
        data_dict = read_file(filename='./data/dataset.p')
        worker = Model(worker_idx=args.idx,num_layers=args.num_layers,model=args.model)
        worker.load_model(dataset_valid=data_dict['valid'])
        old_score = worker.worker.score
        worker.train(dataset_train=data_dict['train'],dataset_valid=data_dict['valid'])
        worker.save_model(save_mode=0)
        new_score = worker.worker.score
        print('Worker-%d score trained from %.5f --> %.5f'%(args.idx,old_score,new_score),flush=True)
        # NOTE: avoid racing
        time.sleep(random.random())
        pickle.dump({args.idx:new_score},open('./%d-%s-population/scores.p'%(args.num_layers,args.model),'ab'))
    elif args.phase == 'explore':
        data_dict = read_file(filename='./data/dataset.p')
        good_worker = Model(worker_idx=args.target_idx,num_layers=args.num_layers,model=args.model)
        good_worker.load_model(dataset_valid=data_dict['valid'])
        worker = Model(worker_idx=args.idx,num_layers=args.num_layers,model=args.model)
        worker.load_model(dataset_valid=data_dict['valid'])
        worker.explore_model(good_model=good_worker.worker.model)
        worker.save_model(save_mode=0)
        print('Worker-%d (%.5f) explore worker-%d (%.5f), initial score = %.5f'%(worker.worker.worker_idx,worker.worker.score,
        good_worker.worker.worker_idx,good_worker.worker.score,worker.worker.score),flush=True)
    elif args.phase == 'conclude':
        data_dict = read_file(filename='./data/dataset.p')
        worker = Model(worker_idx=args.idx,num_layers=args.num_layers,model=args.model)
        worker.load_model(dataset_valid=data_dict['valid'])
        worker.save_model(save_mode=1)
        print('Worker-%d has the best architecture. Score : %.5f'%(worker.worker.worker_idx,worker.worker.score),flush=True)
    elif args.phase == 'won':
        data_dict = read_file(filename='./data/dataset.p')
        worker = Model(worker_idx=args.idx,num_layers=args.num_layers,model=args.model)
        worker.load_model(dataset_valid=data_dict['valid'])
        worker.save_model(save_mode=2)
        print('Worker-%d has won! Score : %.5f'%(worker.worker.worker_idx,worker.worker.score),flush=True)
    else:
        raise Exception('Illegal phase = %s'%args.phase)