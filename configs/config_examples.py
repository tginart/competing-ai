from agent import *
from data import *
from auction import *
from user import *
import argh
import os
import pickle
import copy

def config006EX():
    '''
    '''
    #EXP CONFIGS
    exp = dict()
    exp['expno'] = '006EX'
    exp['n_runs'] = 5*8*7
    
    #RUN CONFIGS
    runs = []
    r_id = 0
    run_temp = dict()
    run_temp['use-gpu'] = False
    run_temp['auction'] = inefficient_classification_auction
    run_temp['auction_args'] = {'c':0, 'r':1}
    run_temp['dataset'] = healthDataset
    run_temp['dataset_init'] = healthDataset
    run_temp['users'] = []
    run_temp['dargs'] = {'path':'/home/datasets/',
                        'csv_prefix' : 'Postures'}
    run_temp['print-freq'] = 100
    run_temp['max-iters'] = 2000
    run_temp['special-log'] = ['agg_corr', 'y_hats']
    for seed in [1,2,3,4,5]:
        for alpha in [0,1,1.5,2,3,4,8]:
            for na in [1,2,4,8,16,32,64,128]:
                run = copy.deepcopy(run_temp)
                run['auction_args']['alpha'] = alpha
                run['seed'] = seed
                run['agents'] = []
                for i in range(na):
                    a = dict()
                    a['class'] = knnAgent
                    a['args'] = {'k':1}
                    a['init_ds'] = 3
                    run['agents'].append(a)
                run['r_id'] = r_id
                r_id += 1
                runs.append(run)
    return exp, runs


def config005EX():
    '''
    EXP NO. 005EX
    Hyp: nn w adult
    official 
    '''
    #EXP CONFIGS
    exp = dict()
    exp['expno'] = '005EX'
    exp['n_runs'] = 5*8*9
   
    #RUN CONFIGS
    runs = []
    r_id = 0
    run_temp = dict()
    run_temp['use-gpu'] = False
    run_temp['auction'] = inefficient_classification_auction
    run_temp['auction_args'] = {'c':0, 'r':1}
    run_temp['dataset'] = healthDataset
    run_temp['dataset_init'] = healthDataset
    run_temp['users'] = []
    run_temp['dargs'] = {'path':'/home/datasets/',
                        'csv_prefix' : 'adult'}
    run_temp['print-freq'] = 100
    run_temp['max-iters'] = 2000
    run_temp['special-log'] = ['agg_corr', 'y_hats']
    for seed in [1,2,3,4,5]:
        for alpha in [0,0.5,0.75,1,1.5,2,3,4,8]:
            for na in [1,2,4,8,16,32,64,128]:
                run = copy.deepcopy(run_temp)
                run['auction_args']['alpha'] = alpha
                run['seed'] = seed
                run['agents'] = []
                for i in range(na):
                    a = dict()
                    a['class'] = knnAgent
                    a['args'] = {'k':1}
                    a['init_ds'] = 3
                    run['agents'].append(a)
                run['r_id'] = r_id
                r_id += 1
                runs.append(run)
    return exp, runs


def config004EX():
    '''
    EXP NO. 004EX
    Hyp: Run experiment with F-MNIST with 3 random seeds
    Easier to rerun this with update spec-logging
    NN on fmnist off
    '''
    #EXP CONFIGS
    exp = dict()
    exp['expno'] = '004EX'
    exp['n_runs'] = 8*8*3
    exp['use_queue'] = True

    #RUN CONFIGS
    runs = []
    r_id = 0
    run_temp = dict()
    run_temp['use-gpu'] = False
    run_temp['auction'] = inefficient_classification_auction
    run_temp['auction_args'] = {'c':0, 'r':1}
    run_temp['dataset'] = fashion_mnist
    run_temp['dataset_init'] = fashion_mnist
    run_temp['users'] = []
    run_temp['dargs'] = {'path':'/home/FashionMNIST'}
    run_temp['print-freq'] = 100
    run_temp['max-iters'] = 10000
    run_temp['special-log'] = ['agg_corr', 'y_hats']
    for seed in [1,2,3]:
        for alpha in [0,0.5,1,1.5,2,3,4,8]:
            for na in [1,2,4,8,16,32,64,128]:
                run = copy.deepcopy(run_temp)
                run['auction_args']['alpha'] = alpha
                run['seed'] = seed
                run['agents'] = []
                for i in range(na):
                    a = dict()
                    a['class'] = knnAgent
                    a['args'] = {'k':1}
                    a['init_ds'] = 100
                    run['agents'].append(a)
                run['r_id'] = r_id
                r_id += 1
                runs.append(run)
    return exp, runs


def config003EX():
    '''
    EXP NO. 003EX
    Hyp: MLP layer  w Postures
    '''
    #EXP CONFIGS
    exp = dict()
    exp['expno'] = '003EX'
    exp['n_runs'] = 5*8*8
    exp['use_queue'] = True


    #RUN CONFIGS
    runs = []
    r_id = 0
    run_temp = dict()
    run_temp['use-gpu'] = False
    run_temp['auction'] = inefficient_classification_auction
    run_temp['auction_args'] = {'c':0, 'r':1}
    run_temp['dataset'] = healthDataset
    run_temp['dataset_init'] = healthDataset
    run_temp['users'] = []
    run_temp['dargs'] = {'path':'home/datasets/',
                        'csv_prefix' : 'Postures'}
    run_temp['print-freq'] = 100
    run_temp['max-iters'] = 4000
    run_temp['special-log'] = ['agg_corr', 'y_hats']
    for seed in [1,2,3,4,5]:
        for alpha in [0,1,1.5,1.75,2,3,4,8]:
            for na in [1,2,4,8,16,32,64,128]:
                run = copy.deepcopy(run_temp)
                run['auction_args']['alpha'] = alpha
                run['seed'] = seed
                run['agents'] = []
                for i in range(na):
                    a = dict()
                    a['class'] = mlpAgent
                    a['args'] = {'x_dim':16, 'n_class':10, 'epoch':32, 
                    'n_layers':1,'lr':1e-3, 'task':"C", 'hidden':16,
                    'retrain_limit':4,'bsize':32,'retrain_max':1*10**3}
                    a['init_ds'] = 3
                    run['agents'].append(a)
                run['r_id'] = r_id
                r_id += 1
                runs.append(run)
    return exp, runs


def config002EX():
    '''
    EXP NO. 002EX
    Hyp: MLP  w adult
    '''
    #EXP CONFIGS
    exp = dict()
    exp['expno'] = '002EX'
    exp['n_runs'] = 5*8*9
    exp['use_queue'] = True

    #RUN CONFIGS
    runs = []
    r_id = 0
    run_temp = dict()
    run_temp['use-gpu'] = False
    run_temp['auction'] = inefficient_classification_auction
    run_temp['auction_args'] = {'c':0, 'r':1}
    run_temp['dataset'] = healthDataset
    run_temp['dataset_init'] = healthDataset
    run_temp['users'] = []
    run_temp['dargs'] = {'path':'/home/datasets/',
                        'csv_prefix' : 'adult'}
    run_temp['print-freq'] = 100
    run_temp['max-iters'] = 4000
    run_temp['special-log'] = ['agg_corr', 'y_hats']
    for seed in [1,2,3,4,5]:
        for alpha in [0,0.5,0.75,1,1.5,2,3,4,8]:
            for na in [1,2,4,8,16,32,64,128]:
                run = copy.deepcopy(run_temp)
                run['auction_args']['alpha'] = alpha
                run['seed'] = seed
                run['agents'] = []
                for i in range(na):
                    a = dict()
                    a['class'] = mlpAgent
                    a['args'] = {'x_dim':50, 'n_class':10, 'epoch':32, 
                    'n_layers':1,'lr':1e-3, 'task':"C", 'hidden':64,
                    'retrain_limit':32,'bsize':32,'retrain_max':1*10**3}
                    a['init_ds'] = 3
                    run['agents'].append(a)
                run['r_id'] = r_id
                r_id += 1
                runs.append(run)
    return exp, runs


def config001EX():
    '''
    EXP NO. 001EX
    Hyp: Run experiment with FMNIST with 3 random seeds
    USE MLP
    '''
    #EXP CONFIGS
    exp = dict()
    exp['expno'] = '001EX'
    exp['n_runs'] = 8*8*3
    exp['use_queue'] = True

    #RUN CONFIGS
    runs = []
    r_id = 0
    run_temp = dict()
    run_temp['use-gpu'] = False
    run_temp['auction'] = inefficient_classification_auction
    run_temp['auction_args'] = {'c':0, 'r':1}
    run_temp['dataset'] = fashion_mnist
    run_temp['dataset_init'] = fashion_mnist
    run_temp['users'] = []
    run_temp['dargs'] = {'path':'/home/FashionMNIST'}
    run_temp['print-freq'] = 100
    run_temp['max-iters'] = 10000
    run_temp['special-log'] = ['agg_corr', 'y_hats']
    for seed in [1,2,3]:
        for alpha in [0,0.5,1,1.5,2,3,4,8]:
            for na in [1,2,4,8,16,32,64,128]:
                run = copy.deepcopy(run_temp)
                run['auction_args']['alpha'] = alpha
                run['seed'] = seed
                run['agents'] = []
                for i in range(na):
                    a = dict()
                    a['class'] = mlpAgent
                    a['class'] = mlpAgent
                    a['args'] = {'x_dim':784, 'n_class':10, 'epoch':30, 
                    'n_layers':1,'lr':1e-4, 'task':"C", 'hidden':400,
                    'retrain_limit':500,'bsize':50,'retrain_max':30*10**3}
                    a['init_ds'] = 100
                    run['agents'].append(a)
                run['r_id'] = r_id
                r_id += 1
                runs.append(run)
    return exp, runs
