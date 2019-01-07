import numpy as np
from collections import deque
import gym
import gym_super_mario_bros
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv, PenalizeDeathEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
SIMPLE_MOVEMENT = SIMPLE_MOVEMENT[1:]
from gym import spaces
from PIL import Image
import cv2

PALETTE_ACTIONS = [['NOP'],
 ['up'],
 ['down'],
 ['left'],
 ['left', 'A'],
 ['left', 'B'],
 ['left', 'A', 'B'],
 ['right'],
 ['right', 'A'],
 ['right', 'B'],
 ['right', 'A', 'B'],
 ['A'],
 ['B'],
 ['A', 'B']
 ]
def _process_frame_mario(frame):
    if frame is not None:           # for future meta implementation
        img = np.reshape(frame, [240, 256, 3]).astype(np.float32)        
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        x_t = cv2.resize(img, (84, 84))
        x_t = np.reshape(x_t, [1, 84, 84])/255.0
        #x_t.astype(np.uint8)

    else:
        x_t = np.zeros((1, 84, 84))
    return x_t



class ProcessFrameMario(gym.Wrapper):
    def __init__(self, env=None, reward_type=None):
        super(ProcessFrameMario, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)
        self.prev_time = 400
        self.prev_stat = 0
        self.prev_score = 0
        self.prev_dist = 40
        self.reward_type = reward_type
        self.milestones = [1000,1500,2000,2500,3000]
        self.counter = 0

    def step(self, action):
        ''' 
            Implementing custom rewards
                Time = -0.1
                Distance = +1 or 0 
                Player Status = +/- 5
                Score = 2.5 x [Increase in Score]
                Done = +50 [Game Completed] or -50 [Game Incomplete]
        '''
        obs, reward, done, info = self.env.step(action)
        
        if self.reward_type == 'sparse':
            reward = 0 
            if (self.counter < 5) and (info['x_pos'] > self.milestones[self.counter])  : 
                reward = 20 
                self.counter = self.counter + 1 
            
            if done : 
                if info['flag_get'] :
                    reward = 50
                else:
                    reward = -50
            
        elif self.reward_type == 'dense':
            print('im here')
            reward = max(min((info['x_pos'] - self.prev_dist - 0.05), 2), -2)
            self.prev_dist = info['x_pos']
            
            reward += (self.prev_time - info['time']) * -0.1
            self.prev_time = info['time']

            reward += (int(info['status']!='small')  - self.prev_stat) * 5
            self.prev_stat = int(info['status']!='small')

            reward += (info['score'] - self.prev_score) * 0.025
            self.prev_score = info['score']

            if done:
                if info['flag_get'] :
                    reward += 500
                else:
                    reward -= 50
                    
        else : return None
        
        return _process_frame_mario(obs), reward, done, info

    def reset(self):
        self.prev_time = 400
        self.prev_stat = 0
        self.prev_score = 0
        self.prev_dist = 40
        return _process_frame_mario(self.env.reset())

    def change_level(self, level):
        self.env.change_level(level)


class BufferSkipFrames(gym.Wrapper):
    def __init__(self, env=None, skip=4, shape=(84, 84)):
        super(BufferSkipFrames, self).__init__(env)
        self.counter = 0
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
        self.skip = skip
        self.buffer = deque(maxlen=self.skip)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        counter = 1
        total_reward = reward
        self.buffer.append(obs)

        for i in range(self.skip - 1):            
            if not done:
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                counter +=1
                self.buffer.append(obs)
            else:
                self.buffer.append(obs)

        frame = np.stack(self.buffer, axis=0)
        frame = np.reshape(frame, (4, 84, 84))
        return frame, total_reward, done, info

    def reset(self):
        self.buffer.clear()
        obs = self.env.reset()
        for i in range(self.skip):
            self.buffer.append(obs)

        frame = np.stack(self.buffer, axis=0)
        frame = np.reshape(frame, (4, 84, 84))
        return frame
    
    def change_level(self, level):
        self.env.change_level(level)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        if observation is not None:    # for future meta implementation
            self.num_steps += 1
            self.state_mean = self.state_mean * self.alpha + \
                observation.mean() * (1 - self.alpha)
            self.state_std = self.state_std * self.alpha + \
                observation.std() * (1 - self.alpha)

            unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
            unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

            return (observation - unbiased_mean) / (unbiased_std + 1e-8)
        
        else:
            return observation

    def change_level(self, level):
        self.env.change_level(level)

def wrap_mario(env, reward_type):
    # assert 'SuperMarioBros' in env.spec.id
    env = ProcessFrameMario(env, reward_type)
    env = NormalizedEnv(env)
    env = BufferSkipFrames(env)
    return env

def create_mario_env(env_id,  reward_type):
    env = gym_super_mario_bros.make(env_id)
    env = BinarySpaceToDiscreteSpaceEnv(env, PALETTE_ACTIONS)
    env = wrap_mario(env, reward_type)
    return env
