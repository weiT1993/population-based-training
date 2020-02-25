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
    parser.add_argument('--phase', type=str, choices=['init','train','explore'])
    parser.add_argument('--target-idx', metavar='N', default=-1, type=int,help='Best Worker Index')
    args = parser.parse_args()

    if args.phase == 'init':
        print('Worker-%d instantiated'%args.idx,flush=True)
        worker = FC(worker_idx=args.idx)
        worker.random_model(num_layers=3)
        worker.save_model()
    elif args.phase == 'train':
        data_dict = read_file(filename='./data/dataset.p')
        worker = FC(worker_idx=args.idx)
        worker.load_model(dataset_valid=data_dict['valid'])
        old_score = worker.score
        worker.train(dataset_train=data_dict['train'],dataset_valid=data_dict['valid'])
        worker.save_model()
        new_score = worker.score
        print('Worker-%d score trained from %.5f --> %.5f'%(args.idx,old_score,new_score),flush=True)
        # NOTE: avoid racing
        time.sleep(random.random())
        pickle.dump({args.idx:new_score},open('./population/scores.p','ab'))
    elif args.phase == 'explore':
        data_dict = read_file(filename='./data/dataset.p')
        good_worker = FC(worker_idx=args.target_idx)
        good_worker.load_model(dataset_valid=data_dict['valid'])
        worker = FC(worker_idx=args.idx)
        worker.load_model(dataset_valid=data_dict['valid'])
        old_score = worker.score
        worker.explore_model(good_model=good_worker.model,dataset_valid=data_dict['valid'])
        worker.save_model()
        print('Worker-%d (%.5f) explore worker-%d (%.5f), initial score = %.5f'%(worker.worker_idx,old_score,
        good_worker.worker_idx,good_worker.score,worker.score),flush=True)