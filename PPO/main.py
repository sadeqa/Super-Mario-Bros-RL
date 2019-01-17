import copy
import glob
import os
import time
from collections import deque
import csv

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.icm import ICM
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from a2c_ppo_acktr.visualize import visdom_plot


args = get_args()

norm_pos = 3161

use_curiosity = args.use_curiosity

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    
folder = 'default_' if not use_curiosity else 'curs_'
save_folder = args.log_dir + folder + args.reward_type

train_file = save_folder + '/train_reward.csv'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
        
eval_folder = save_folder + "/_eval"
if not os.path.exists(eval_folder):
    os.makedirs(eval_folder)
eval_file = eval_folder + '/eval_reward.csv'


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    print(device)
    print(save_folder)

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False, args.reward_type)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space)
    actor_critic.to(device)
    
    curiosity = None
    if use_curiosity :
        curiosity = ICM(envs.observation_space.shape[0], envs.action_space.n)
        curiosity.to(device)
    
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, curiosity, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,max_grad_norm=args.max_grad_norm, use_curiosity=use_curiosity)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=args.num_processes * 2)

    start = time.time()
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            if args.algo == "acktr":
                # use optimizer's learning rate since it's hard-coded in kfac.py
                update_linear_schedule(agent.optimizer, j, num_updates, agent.optimizer.lr)
            else:
                update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_clip_decay:
            agent.clip_param = args.clip_param  * (1 - j / float(num_updates))

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = agent.actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            envs.render()
            
            cur_reward = reward
            
            if use_curiosity :
                action_one_hot = (torch.eye(14)[action]).view(-1,14).cuda()
                _, pred_phi, actual_phi = curiosity((rollouts.obs[step], obs, action_one_hot))
                cur_reward += ((pred_phi - actual_phi).pow(2)).sum(-1, keepdim=True).cpu()/20
            
            for i, finished in enumerate(done):
                if finished:
                    percentile = infos[i]['x_pos']/norm_pos
                    episode_rewards.append(percentile)
                    with open(train_file[:-4] + str(i) + train_file[-4:], 'a', newline='') as sfile:
                        writer = csv.writer(sfile)
                        writer.writerows([[percentile]])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, cur_reward.detach(), masks)

        with torch.no_grad():
            next_value = agent.actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = agent.actor_critic
            if args.cuda:
                save_model = copy.deepcopy(agent.actor_critic).cpu()

            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]

            torch.save(save_model, os.path.join(save_folder, '/' + args.env_name + ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > args.num_processes:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards), dist_entropy,
                       value_loss, action_loss))
#Evaluation time :

        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            
            num_proc = 1
            eval_envs = make_vec_envs(
                args.env_name, args.seed + num_proc, num_proc,
                args.gamma, args.log_dir, args.add_timestep, device, True, args.reward_type)

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(num_proc,
                            actor_critic.recurrent_hidden_state_size, device=device)
            
            eval_masks = torch.zeros(num_proc, 1, device=device)
            positions = deque(maxlen=400)

            while len(eval_episode_rewards) < 1:
                with torch.no_grad():
                    
                    _, action, _, eval_recurrent_hidden_states = agent.actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)
                eval_envs.render()

                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done]).cuda()

                for i, finished in enumerate(done):
                    if finished:
                        percentile = infos[i]['x_pos']/norm_pos
                        eval_episode_rewards.append(percentile)
                        with open(eval_file, 'a', newline='') as sfile:
                            writer = csv.writer(sfile)
                            writer.writerows([[percentile]])
                
                #to prevent the agent from getting stuck
                positions.append(infos[0]['x_pos'])
                pos_ar = np.array(positions)
                if (len(positions) >= 200) and (pos_ar < pos_ar[-1] + 20).all() and (pos_ar > pos_ar[-1] - 20).all():
                    percentile = infos[0]['x_pos']/norm_pos
                    eval_episode_rewards.append(percentile)
                    with open(eval_file, 'a', newline='') as sfile:
                        writer = csv.writer(sfile)
                        writer.writerows([[percentile]])
                    
            eval_envs.close()
            positions.clear()

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                format(len(eval_episode_rewards),
                       np.mean(eval_episode_rewards)))
            

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  args.algo, args.num_env_steps)
            except IOError:
                pass


if __name__ == "__main__":
    main()
