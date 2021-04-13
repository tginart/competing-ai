from auction import *
from user import *
from analysis.analysis_utils import _y_hats, _agg_corr
import dill as pickle
import torch
import torchvision
import torch.utils 
import random
import numpy as np
import argh
import warnings
from collections import defaultdict

parser= argh.ArghParser()

def setup_agents(config):
    _set_seed(config)
    agents = []
    for i,a in enumerate(config['agents']):

        #_set_seed(config,mod=i)
        agent = a['class'](**a['args'])
        agent.id = i  
        agents.append(agent)
    return agents

def setup_users(config):
    
    users = []
    for i,u in enumerate(config['users']):
        
        user = u['class'](**u['args'])
        user.id = i  
        users.append(user)
    return users

def setup_users_special(config, agents, datastream):
    # a routine for setting up users with special args
    for user in datastream.users:
        if type(user) == perfect_user:
            user.setup_learners(agents)

def setup_mechanism(config, agents, datastream):
    return config['auction'](agents,datastream,
                            **config['auction_args'])
        
def init_agent_data(config, agents):

    '''
    if config['dataset_init'] is not None:
        data_loader = torch.utils.data.DataLoader(
            config['dataset_init'](**config['dargs']),
            batch_size=config['agents'][j]['init_ds'],
            shuffle=True)
        j = 0
        c = 0
        for i, batch in enumerate(data_loader):
            x,y = batch
            #x = x.reshape(-1,1)
            agents[j].add_data(x,y)
            j +=1
            if j >= len(agents):
                break
            
            c += 1
            if c >= config['agents'][j]['init_ds']:
                c = 0
                j += 1
                if j >= len(agents):
                    break
        '''
    #apply training on seed data, if needed
    for j,a in enumerate(agents):

        _set_seed(config,mod=j+1)

        #define config if not none
        if config['dataset_init'] is not None:
            data_loader = torch.utils.data.DataLoader(
                config['dataset_init'](**config['dargs']),
                batch_size=config['agents'][j]['init_ds'],
                shuffle=True)

            for _,batch in enumerate(data_loader): #only run once
                x,y = batch
                a.add_data(x,y)
                break
            a._seed_train() 
    #print(agents[0].X == agents[1].X)
         


def setup_datastream(config):
    _set_seed(config,mod=-config['seed'])
    return torch.utils.data.DataLoader(
        config['dataset'](**config['dargs']),
        batch_size=1,
        shuffle=True)

def setup_log(config, agents):
    logger = defaultdict(list)
    #need to define lambdas in order to pickle
    logger['agents'] = defaultdict(lambda : defaultdict(list))
    logger['config'] = config
    
    for i,a in enumerate(agents):
        assert a.id == i, "Agent id mismatch in logging setup"
        
    return logger

def _set_seed(config,mod=0):
    torch.manual_seed(config['seed']+mod)
    random.seed(config['seed']+mod)
    np.random.seed(config['seed']+mod)


def _special_log(config,logger):
    special_log = dict()
    if 'special-log' in config:
        for sl in config['special-log']:
            if sl == 'y_hats':
                special_log['y_hats'] = _y_hats(logger)
            elif  sl == 'agg_corr':
                special_log['agg_corr'] = _agg_corr(logger)
            else:
                warnings.warn(f'Special logging of type {sl} not supported')
    np.savez(open('special_log','wb'),**special_log) 
    return special_log


def _sim(config, auction, datastream, logger):
    _set_seed(config) 
    run_id = config['r_id'] if 'r_id' in config else ''

    for i,data in enumerate(datastream):
        auction.logger = logger 
        #TODO: eventually convert logging to self.logger as above in auction
        auction.run(data, logger)
        if i % config['print-freq'] == 0:
            print(f'Run{run_id}: At round {i}',flush=True)
            #print(np.mean(logger['agents']['agg-correct']))

        if i >= config['max-iters']:
            print(f'Run{run_id}: simulation complete')
            break
    

def main(config):
    agents = setup_agents(config)
    init_agent_data(config, agents)
    datastream = setup_datastream(config)
    datastream.users = setup_users(config)
    auction = setup_mechanism(config, agents, datastream)
    setup_users_special(config, agents, datastream)
    logger = setup_log(config, agents)
    _sim(config, auction, datastream, logger)
    _special_log(config, logger)
    pickle.dump(logger, open('logger.pickle','wb'))


parser.add_commands([main])
# dispatching 
if __name__ == '__main__':
    parser.dispatch()

