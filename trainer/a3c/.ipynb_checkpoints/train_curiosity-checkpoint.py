import os
import time
from collections import deque
import csv

import numpy as np
import cv2
from itertools import count
from gym import wrappers

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from models.actor_critic import ActorCritic
from models.icm import ICM

from common.atari_wrapper import create_mario_env
from common.mario_actions import ACTIONS

cross_entropy = torch.nn.CrossEntropyLoss()

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, shared_curiosity, counter, lock, optimizer=None, select_sample=True):
    torch.manual_seed(args.seed + rank)

    print("Process No : {} | Sampling : {}".format(rank, select_sample))

    FloatTensor = torch.FloatTensor# torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
    DoubleTensor = torch.DoubleTensor# torch.cuda.DoubleTensor if args.use_cuda else torch.DoubleTensor
    ByteTensor = torch.ByteTensor# torch.cuda.ByteTensor if args.use_cuda else torch.ByteTensor

    savefile = os.getcwd() + '/save/curiosity_'+ args.reward_type +'/train_reward.csv'
    saveweights = os.getcwd() + '/save/curiosity_'+ args.reward_type +'/mario_a3c_params.pkl'

    env = create_mario_env(args.env_name, args.reward_type)
    #env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], len(ACTIONS))
    if optimizer is None:
        optimizer = optim.Adam(list(shared_model.parameters()) + list(shared_curiosity.parameters()), lr=args.lr)
        
    curiosity = ICM(env.observation_space.shape[0], len(ACTIONS))
#     cur_optimizer = optim.Adam(shared_curiosity.parameters(), lr=args.lr)
    
    model.train()
    curiosity.train()

    state = env.reset()
    cum_rew = 0 
    state = torch.from_numpy(state)
    done = True
    
    episode_length = 0
    for num_iter in count():
        #env.render()
        if rank == 0:
            
            if num_iter % args.save_interval == 0 and num_iter > 0:
                print ("Saving model at :" + saveweights)            
                torch.save(shared_model.state_dict(), saveweights)
                torch.save(shared_curiosity.state_dict(), saveweights[:-4] + '_curiosity.pkl')

        if num_iter % (args.save_interval * 2.5) == 0 and num_iter > 0 and rank == 1:    # Second saver in-case first processes crashes 
            print ("Saving model for process 1 at :" + saveweights)            
            torch.save(shared_model.state_dict(), saveweights)
            torch.save(shared_curiosity.state_dict(), saveweights[:-4] + '_curiosity.pkl')
            
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        curiosity.load_state_dict(shared_curiosity.state_dict())
        
        if done:
            cx = Variable(torch.zeros(1, 512)).type(FloatTensor)
            hx = Variable(torch.zeros(1, 512)).type(FloatTensor)
        else:
            cx = Variable(cx.data).type(FloatTensor)
            hx = Variable(hx.data).type(FloatTensor)

        values = []
        log_probs = []
        rewards = []
        entropies = []
        forward_losses = []
        inverse_losses = []
        #reason =''
        
        for step in range(args.num_steps):
            episode_length += 1            
            state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)
            value, logit, (hx, cx) = model((state_inp, (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(-1, keepdim=True)
            entropies.append(entropy)
            
            
            if select_sample:
                action = prob.multinomial(1).data
            else:
                action = prob.max(-1, keepdim=True)[1].data
                
            log_prob = log_prob.gather(-1, Variable(action))
            
            action_out = int(action[0, 0].data.numpy())
            state, reward, done, _ = env.step(action_out)
            cum_rew = cum_rew + reward
            
            action_one_hot = (torch.eye(len(ACTIONS))[action_out]).view(1,-1)
            
            next_state_inp = Variable(torch.from_numpy(state).unsqueeze(0)).type(FloatTensor)
            logits_pred, pred_phi, actual_phi = curiosity((state_inp, next_state_inp, action_one_hot))
            
            inverse_loss = cross_entropy(logits_pred, action[0])/len(ACTIONS)
            forward_loss = ((pred_phi - actual_phi).pow(2)).sum(-1, keepdim=True)/2
            
            done = done or episode_length >= args.max_episode_length
            
            int_reward = (args.eta*forward_loss).data.numpy()[0,0]
            
            reward = int_reward + reward/10
            reward = max(min(reward, 500), -50)
            
            
            with lock:
                counter.value += 1

            if done:
                episode_length = 0
#                 env.change_level(0)
                state = env.reset()
                with open(savefile[:-4]+'_{}.csv'.format(rank), 'a', newline='') as sfile:
                    writer = csv.writer(sfile)
                    writer.writerows([[cum_rew]])
                cum_rew = 0 
 #               print ("Process {} has completed.".format(rank))

#            env.locked_levels = [False] + [True] * 31
            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            forward_losses.append(forward_loss)
            inverse_losses.append(inverse_loss)
            
            
            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)
            value, _, _ = model((state_inp, (hx, cx)))
            R = value.data

        values.append(Variable(R).type(FloatTensor))
        policy_loss = 0
        value_loss = 0
        curiosity_loss = 0
        R = Variable(R).type(FloatTensor)
        gae = torch.zeros(1, 1).type(FloatTensor)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - log_probs[i] * Variable(gae).type(FloatTensor) - args.entropy_coef * entropies[i]
            
            curiosity_loss += (1 - args.beta)*inverse_losses[i] + args.beta*forward_losses[i]
            
        total_loss = args.lambd*(policy_loss + args.value_loss_coef * value_loss)
        
#        print ("Process {} loss :".format(rank), total_loss.data)
        optimizer.zero_grad()
#         cur_optimizer.zero_grad()
        
        (total_loss + 10.0*curiosity_loss).backward()
#         (curiosity_loss).backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(curiosity.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        ensure_shared_grads(curiosity, shared_curiosity)
        
        optimizer.step()
        
#    print(rank)
#    print ("Process {} closed.".format(rank))

def test(rank, args, shared_model, counter):
    torch.manual_seed(args.seed + rank)

    FloatTensor = torch.FloatTensor# torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
    DoubleTensor = torch.DoubleTensor# torch.cuda.DoubleTensor if args.use_cuda else torch.DoubleTensor
    ByteTensor = torch.ByteTensor# torch.cuda.ByteTensor if args.use_cuda else torch.ByteTensor

    env = create_mario_env(args.env_name, args.reward_type)
    """ 
        need to implement Monitor wrapper with env.change_level
    """
    # expt_dir = 'video'
    # env = wrappers.Monitor(env, expt_dir, force=True, video_callable=lambda count: count % 10 == 0)
    
    #env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], len(ACTIONS))
    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True
    savefile = os.getcwd() + '/save/curiosity_'+ args.reward_type +'/mario_curves.csv'
    
    title = ['Time','No. Steps', 'Total Reward', 'Episode Length']
    with open(savefile, 'a', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)    

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=400)
    positions = deque(maxlen=400)
    episode_length = 0
    while True:
        episode_length += 1
        ep_start_time = time.time()
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 512),  requires_grad=True ).type(FloatTensor)
            with torch.no_grad():
                cx=cx
            hx = Variable(torch.zeros(1, 512),  requires_grad=True).type(FloatTensor)
            with torch.no_grad():
                hx=hx

        else:
            with torch.no_grad():
                cx = Variable(cx.data).type(FloatTensor)
                hx = Variable(hx.data).type(FloatTensor)
        

        with torch.no_grad(): state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)
        value, logit, (hx, cx) = model((state_inp, (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(-1, keepdim=True)[1].data
        action_out = int(action[0, 0].data.numpy())
        state, reward, done, info = env.step(action_out)
        #env.render()
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True
        if args.pos_stuck :
            positions.append(info['x_pos'])
            pos_ar = np.array(positions)
            if (len(positions) >= 200) and (pos_ar < pos_ar[-1] + 20).all() and (pos_ar > pos_ar[-1] - 20).all():
                done = True

        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)), 
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            
            data = [time.time() - ep_start_time,
                    counter.value, reward_sum, episode_length]
            
            with open(savefile, 'a', newline='') as sfile:
                writer = csv.writer(sfile)
                writer.writerows([data])
            
            reward_sum = 0
            episode_length = 0
            actions.clear()
            time.sleep(60)
#             env.locked_levels = [False] + [True] * 31
#             env.change_level(0)
            state = env.reset()

        state = torch.from_numpy(state)
