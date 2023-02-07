# %%
import os
import sys
sys.path.append('../')

import argparse
import numpy as np
import copy
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import my_optim

import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt

from gym.core import ObservationWrapper
from gym.spaces.box import Box
from PIL import Image
from IPython.display import clear_output
from tqdm import trange

from material.atari_util import *
from material.atari_wrapper import *

parser = argparse.ArgumentParser(description='Pytorch A3C PongDeterministic')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--TD-lambda', type=float, default=1.00,
                    help='lambda parameter for TD (default: 1.00)')
parser.add_argument('--actor-correction',type=float,default=0.01,
                    help='actor loss corection coefficient')
parser.add_argument('--entropy-coef', type=float, default=0.001,
                    help='entropy term coefficient (default: 0.001)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed (default: 123)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=5,
                    help='number of forward steps in A3C (default: 5)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    help='environment to train on (default: PongDeterministic-v4)')
#parser.add_argument('--no-shared', default=False,
#                    help='use an optimizer without shared momentum.')

# %%
# GPU 장치는 A3C에 적용할수 없습니다.
device = torch.device('cuda')

# %%
"""
# Pong environment preprocess
"""

# %%
class PreprocessAtariObs(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.img_size = (1, 64, 64)
        self.observation_space = Box(0.0, 1.0, self.img_size,dtype=np.float32)

    def _to_gray_scale(self, rgb, channel_weights=[0.7, 0.1, 0.2]):
        dummy = 0
        for idx,channel_weight in enumerate(channel_weights):
            dummy += channel_weight*(rgb[:,:,idx])
        return np.expand_dims(dummy,axis=-1)
    
    def observation(self, img):      
        img = img[34:34+160, :160]
        img = Image.fromarray(np.uint8(img),'RGB')
        img = img.resize((64,64))
        img = np.array(img)
        img = self._to_gray_scale(img)/255.
        return np.array(img,dtype=np.float32).transpose((2,0,1))


def PrimaryAtariWrap(env,clip_rewards=True):
    env = ClipRewardEnv(env)
    env = PreprocessAtariObs(env)    # 이미지 전처리
    return env

# %%
class FrameBuffer(Wrapper):
    def __init__(self, env, n_frames=4):
        """A gym wrapper that reshapes, crops and scales image into the desired shapes"""
        super(FrameBuffer, self).__init__(env)     
        n_channels, height, width = env.observation_space.shape
        obs_shape = [n_channels * n_frames, height, width]
        self.observation_space = Box(0.0, 1.0, obs_shape, dtype=np.float32)
        self.framebuffer = np.zeros(obs_shape)

    def reset(self):
        """resets breakout, returns initial frames"""
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(self.env.reset())
        return self.framebuffer

    def step(self, action):
        """plays breakout for 1 step, returns frame buffer"""
        new_img, reward, done, info = self.env.step(action)
        self.update_buffer(new_img)
        return self.framebuffer, reward, done, info

    def update_buffer(self, img):
        offset = self.env.observation_space.shape[0]
        cropped_framebuffer = self.framebuffer[:-offset]
        self.framebuffer = np.concatenate(
            [img, cropped_framebuffer], axis=0)

# %%
#def make_env(clip_rewards=True, seed=None):
#    env = gym.make('PongDeterministic-v4') 
#    if seed is not None:
#        env.seed(seed)
#    env = PrimaryAtariWrap(env, clip_rewards)
#    env = FrameBuffer(env, n_frames=4)
#    return env
def make_env(clip_rewards=True, seed=None):
    env = gym.make('CartPole-v0') 
    if seed is not None:
        env.seed(seed)
    return env

# %%
class A3C_Agent(nn.Module):
    def __init__(self,num_actions):
        super(A3C_Agent,self).__init__()
        #self.seq = nn.Sequential(
        #    nn.Conv2d(4,32,5,stride=2),
        #    nn.ReLU(),
        #    nn.Conv2d(32,64,5,stride=2),
        #    nn.ReLU(),
        #    nn.Conv2d(64,64,5,stride=2),
        #    nn.ReLU(),
        #    nn.Flatten(),
        #    nn.Linear(1600,256),
        #    nn.ReLU(),
        #)
        self.seq = nn.Sequential(
            nn.Linear(4,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU()
        )
        
        self.policy = nn.Linear(64,num_actions)
        self.value = nn.Linear(64,1)

    def forward(self, state_t):
        '''
        입력인자
            state_t : 상태([batch,state_shape]), torch.tensor
        출력인자
            policy : 정책([batch,n_actions]), torch.tensor
            value : 가치함수([batch]), torch.tensor
        '''
        policy = self.policy(self.seq(state_t))    
        #value = self.value(self.seq(state_t)).squeeze(dim=-1)
        value = self.value(self.seq(state_t))
        return policy, value
    
    def sample_actions(self,state_t):
        '''
        입력인자
            state_t : 상태([1,state_shape]), torch.tensor
        출력인자
            action_t : 행동함수 using torch.multinomial
        '''
        policy, _ = self.forward(state_t)
        policy = torch.squeeze(policy)
        softmax_policy = F.softmax(policy,dim=0)
        action = torch.multinomial(softmax_policy, num_samples=1).item()
        return action

def target_return(rewards,next_value,done,gamma=0.99):
    '''
    Calculate G
        G = r_{t1}+gamma*r_{t2}+gamma**2*r_{t3}...+gamma**n*r_{tn+1}+gamma**{n+1}*V(S_{tn})
    '''
    if done:
        running_add = 0
    else:
        running_add = next_value
    discounted_r = np.zeros_like(rewards)
    for idx in reversed(range(len(rewards))):
        running_add = rewards[idx]+gamma*running_add
        discounted_r[idx] = running_add

    return torch.Tensor(discounted_r).to(device)

# %%
def A3C_loss(transition,train_agent,env,gamma=0.99,epsilon=1e-02):
    '''
    A3C loss함수 계산코드
    입력인자
        batch_sample - 리플레이로부터 받은 샘플(S,A,R,S',done)
        train_agent - 훈련에이전트
        env - 환경
        gamma - 할인율
        epsilon - entropy 가중치
    출력인자
        Total_loss
    목적함수 
        -log(policy)*advantage + (value_infer-value_target)**2 + policy*log(policy)
        Actor-loss(exploitation): "log(policy)*advantage"
        Actor-entropy(exploration): "policy*log(policy)"
        Critic-loss: "MSE(value_infer - value_target)"
    '''
    states,actions,rewards,next_states,dones = transition
    
    #states = torch.Tensor([states[0]]).to(device)
    states = torch.Tensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.Tensor(rewards).to(device)
    #next_state = torch.Tensor([next_states[-1]]).to(device)
    next_states = torch.Tensor(next_states).to(device)
    #dones = torch.BoolTensor(dones).to(device)
    policies, values = train_agent(states)
    
    _, next_values = train_agent(next_states)
    #next_values[dones] = 0.

    probs = F.softmax(policies,dim=-1)
    logprobs = F.log_softmax(policies,dim=-1)
    logp_actions = logprobs[np.arange(states.shape[0]),actions]

    entropy = -torch.sum(probs*logprobs,dim=-1)
    
    # Calculate loss function from backstep
    R = 0
    if dones[-1] != False:
        R = next_values[-1].item()
    values = torch.cat((values,torch.Tensor([[R]]).to(device)),dim=0)

    actor_loss = 0
    critic_loss = 0
    advantage, gen_delta = 0, 0
    for i in reversed(range(len(rewards))):
        R = gamma * R + rewards[i]
        advantage = R - values[i]
        critic_loss +=  advantage.pow(2)

        # Generalized Advantage Estimation
        #delta_t = rewards[i]+gamma*values[i + 1]-values[i]
        delta_t = rewards[i]+gamma*values[i+1]-values[i]
        gen_delta = gamma*gen_delta+delta_t
        actor_loss -= logp_actions[i]*gen_delta.detach()+epsilon*entropy[i]
            #print(log_probs[i].shape, entropies[i].shape)
    
    #print(probs.shape, logprobs.shape, entropy.shape) # [20,6], [20,6], [20]

    #next_value = next_values[-1].item()
    #done = dones[-1]
    #if done:
    #    next_value = 0
    #rewards.append(next_value)
    #values = torch.cat((values,next_value),0)

    #TD_delta = 1
    #actor_loss, critic_loss = 0,0
    #target = next_value
    #for idx in reversed(range(len(rewards))):
    #    target = rewards[idx] + gamma*target
    #    advantages = target - values[idx]
    #    critic_loss += advantages**2
    #    
    #    # TD(lambda) let lambda = 1
    #    #delta = rewards[idx] + gamma*values[idx+1]-values[idx]
    #    #TD_delta = TD_delta*gamma + delta
    #    #actor_loss -= logprobs[idx]*TD_delta 
    #    actor_loss -= logprobs[idx]*advantages.detach()+epsilon*entropy[idx]

    #actor_loss = torch.mean(actor_loss)
    #actor_loss /= len(rewards)
    #critic_loss /= len(rewards)   

    #target_values = target_return(rewards,next_value,done,gamma).view(-1,1)
    #advantages = target_values - values[0]
    #print(logprobs.shape,target_values.shape,advantages.shape,values.shape)
    #target_value = rewards.view(-1,1) + gamma*next_values
    #advantage = target_value-values
    #print(f'target_value: {target_value.shape} value: {values.shape} next_vlaue: {next_values.shape}')
    #actor_loss = -torch.mean(logprobs*advantages.detach()) -epsilon*torch.mean(entropy)
    
    #actor_loss = -(logp_actions*advantage.detach()).mean() - epsilon*torch.mean(entropy).mean()
    #critic_loss = F.mse_loss(target_value.detach(),values)
    
    total_loss = actor_loss + critic_loss
    return total_loss, actor_loss, critic_loss

def train(rank,args,shared_agent):
    env = make_env(True,args.seed+rank)
    optimizer = optim.Adam(shared_agent.parameters(),lr=args.lr)
    
    A3C_record = []
    reward_record = [] 
    start = time.time()
    for ep in range(args.max_episode_length):
        done = False
        state = env.reset()
        total_reward = 0 

        while not done:
            states,actions,rewards,next_states,dones = [],[],[],[],[]
            for step in range(args.num_steps):
                torch_state = torch.Tensor(state).to(device)
                torch_state = torch.unsqueeze(torch_state,0)
                action = shared_agent.sample_actions(torch_state)
                next_state,reward,done,_ = env.step(action)
                total_reward += reward
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                # 업데이트
                state = next_state
                if done:
                    if total_reward == 180:
                        best_agent = copy.deepcopy(shared_agent)
                    break
            transition = (states,actions,rewards,next_states,dones)
            loss,actor_loss,critic_loss = A3C_loss(transition,shared_agent,env,args.gamma,args.entropy_coef)
            #loss *= args.actor_correction
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(shared_agent.parameters(),args.max_grad_norm)
            optimizer.step()

        reward_record.append(total_reward)
        finish = time.time()-start
        if rank == 0 and ep % 100==0:
            A3C_record.append([finish,total_reward]) 

        # Training log
        print(f'|프로세스|{rank}|   >>>   Reward/Episode: {total_reward}/{ep}   Time: {time.strftime("%Hh %Mm %Ss",time.gmtime(finish))}') 
        if np.mean(reward_record[-3:]) >= 180 and total_reward >= 20:
            best_agent = copy.deepcopy(shared_agent)
            print(f" 프로세스: {rank}   >>>   보상: {np.mean(reward_record[-3:])}")
            print(f"학습종료")
            if rank == 1:
                np.save('./A3C_timeVSreward.npy',np.array(A3C_record))
            break 

def test(rank,args,shared_agent,test_games=3): 
    env = make_env(True,args.seed+rank)

    reward_record = [] 
    for ep in range(test_games):
        done = False
        state = env.reset()
        total_reward = 0 

        while True:
            torch_state = torch.Tensor(state).to(device)
            torch_state = torch.unsqueeze(torch_state,0)
            action = shared_agent.sample_actions(torch_state)
            next_state,reward,done,_ = env.step(action)
            total_reward += reward
            
            # 업데이트
            state = next_state
            if done:
                break

        reward_record.append(total_reward)
    print(f'{test_games} 게임')
    print(f'        >> 평균보상: {np.mean(reward_record)}')
    print(f'        >> 최대보상: {np.max(reward_record)}')

if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mp.set_start_method('spawn',force=True)
   
    env = make_env() 
    n_actions = env.action_space.n
    train_agent = A3C_Agent(n_actions).to(device)
    #train_agent.share_memory() # gradients are allocated lazily, so they are not shared here

    train(0,args,train_agent)
    test(0,args,train_agent,3) 
    #torch.save(train_agent,'./ckpt/PongDeterministic/Pong_best_agent_A2C.pth')

# %%
