import argh
import os
import dill as pickle
from configs.config_examples import *
from simulator import main

parser = argh.ArghParser()

def run(exp_id='', run_id=0, runpath=''):
    _,runs = eval(f'config{exp_id}()')
    config = runs[run_id]
    if runpath != '':
        os.chdir(runpath)
    pickle._dump(config, open('config.pickle', 'wb'))
    main(config)

parser = argh.ArghParser()
parser.add_commands([run])


if __name__ == '__main__':
    parser.dispatch()


