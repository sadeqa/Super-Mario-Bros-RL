import os
import argparse
import gym
import numpy as np
import torch
import torch.cuda
import torch.multiprocessing as _mp

from models.actor_critic import ActorCritic
from common.atari_wrapper import create_mario_env
from optimizer.sharedadam import SharedAdam
from trainer.a3c.train import train, test
from common.mario_actions import ACTIONS


SAVEPATH = os.getcwd() + '/save/mario_a3c_params.pkl'

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='discount factor for rewards (default: 0.9)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=250,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 4)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=50,
                    help='number of forward steps in A3C (default: 50)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='SuperMarioBrosNoFrameskip-1-1-v0',
                    help='environment to train on (default: SuperMarioBrosNoFrameskip-1-1-v0)')
parser.add_argument('--no-shared', type=bool, default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--save-interval', type=int, default=10,
                    help='model save interval (default: 10)')
parser.add_argument('--save-path',default=SAVEPATH,
                    help='model save interval (default: {})'.format(SAVEPATH))
parser.add_argument('--non-sample', type=int,default=2,
                    help='number of non sampling processes (default: 2)')
parser.add_argument('--reward_type', type=str, default='basic',
                    help='define the reward type (default: basic)')
parser.add_argument('--pos_stuck', type= bool, default=False,
                    help='penalise getting stuck in a position (default: False)')

mp = _mp.get_context('spawn')

print("Cuda: " + str(torch.cuda.is_available()))

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
        
    args = parser.parse_args()
    env = create_mario_env(args.env_name, args.reward_type)    

    shared_model = ActorCritic(
        env.observation_space.shape[0], len(ACTIONS))

    shared_model.share_memory()
    
    if os.path.isfile(args.save_path):
        print('Loading A3C parametets ...')
        shared_model.load_state_dict(torch.load(args.save_path))

    torch.manual_seed(args.seed)

    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()

    print ("No of available cores : {}".format(mp.cpu_count())) 

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
    
    p.start()
    processes.append(p)

    num_procs = args.num_processes
    no_sample = args.non_sample
   
    if args.num_processes > 1:
        num_procs = args.num_processes - 1    

    sample_val = num_procs - no_sample


    for rank in range(0, num_procs):
        if rank < sample_val:                           # select random action
            p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        else:                                           # select best action
            p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer, False))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
