from models import FC
from utils.helper_fun import read_file

data_dict = read_file(filename='./data/dataset.p')

worker = FC(worker_idx=1)
worker.random_model(num_layers=3)
print('Random model score =',worker.score)
worker.train(dataset_train=data_dict['train'],dataset_valid=data_dict['valid'])
worker.save_model(save_mode=0)
print('Train =',worker.score)

worker = FC(worker_idx=1)
worker.load_model(dataset_valid=data_dict['valid'])
print('Load =',worker.score)
worker.train(dataset_train=data_dict['train'],dataset_valid=data_dict['valid'])
print('Train =',worker.score)

bad_worker = FC(worker_idx=2)
bad_worker.random_model(num_layers=3)
print('Bad random model score =',bad_worker.score)
bad_worker.explore_model(good_model=worker.model,dataset_valid=data_dict['valid'])
print('Bad explore model score =',bad_worker.score)
bad_worker.train(dataset_train=data_dict['train'],dataset_valid=data_dict['valid'])
print('Train =',bad_worker.score)