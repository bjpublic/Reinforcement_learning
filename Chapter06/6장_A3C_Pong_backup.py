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

torch.manual_seed(123)
np.random.seed(123)

# %%
# GPU 장치는 A3C에 적용할수 없습니다.
device = torch.device('cpu')

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
        img = img.resize((42,42))
        img = np.array(img)
        img = self._to_gray_scale(img)/255.
        return np.array(img,dtype=np.float32).transpose((2,0,1))



def PrimaryAtariWrap(env,clip_rewards=True):
    env = MaxAndSkipEnv(env, skip=4) # 설명 : 지나치게 빠른 프레임 -> 4 프레임씩 자르기
    env = EpisodicLifeEnv2(env)       # 설명 라이프 모두 소진시 에피소드 종료
    env = FireResetEnv(env)          # 설명 : 라이프 소진시 자동으로 새 게임 시작(발사)
    if clip_rewards:                 # 설명: 보상에 대한 clipping, 처벌: -1, 보상: +1 로 reward 범위 제한
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
def make_env(clip_rewards=True, seed=None):
    env = gym.make('PongDeterministic-v4') 
    if seed is not None:
        env.seed(seed)
    env = PrimaryAtariWrap(env, clip_rewards)
    env = FrameBuffer(env, n_frames=4)
    return env

# %%
class A3C_Agent(nn.Module):
    def __init__(self,num_actions):
        super(A3C_Agent,self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(4,32,5,stride=2),
            nn.ReLU(),
            nn.Conv2d(32,64,5,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,5,stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1600,256),
            nn.ReLU(),
        )
        self.policy = nn.Linear(256,num_actions)
        self.value = nn.Linear(256,1)

    def forward(self, state_t):
        '''
        입력인자
            state_t : 상태([batch,state_shape]), torch.tensor
        출력인자
            policy : 정책([batch,n_actions]), torch.tensor
            value : 가치함수([batch]), torch.tensor
        '''
        policy = self.policy(self.seq(state_t))    
        value = self.value(self.seq(state_t)).squeeze(dim=-1)
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

# %%
def A2C_loss(transition,train_agent,env,epsilon=1e-03,gamma=0.99):
    '''
    A2C loss함수 계산코드
    입력인자
        batch_sample - 리플레이로부터 받은 샘플(S,A,R,S',done)
        train_agent - 훈련에이전트
        env - 환경
        gamma - 할인율
    출력인자
        Total_loss
    목적함수 
        -log(policy)*advantage + (value_infer-value_target)**2 + policy*log(policy)
        Actor-loss(exploitation): "log(policy)*advantage"
        Actor-entropy(exploration): "policy*log(policy)"
        Critic-loss: "MSE(value_infer - value_target)"
    '''
    states,actions,rewards,next_state,done = transition
    
    states = torch.Tensor(states[None]).to(device)
    #actions = torch.Tensor(actions).to(device).view(-1,num_action)
    rewards = torch.Tensor(rewards[None]).to(device)
    next_state = torch.Tensor(next_state[None]).to(device)
    policies, values = train_agent(states)
    _, next_value = train_agent(next_state)
    if done:
        next_value = 0
    
    probs = F.softmax(policies,dim=-1)
    logprobs = F.log_softmax(policies,dim=-1)

    target_values = rewards+gamma*next_value
    target_values = target_values.squeeze(dim=-1)
    advantages = target_values - values
    entropy = -torch.sum(probs*logprobs,dim=-1)

    actor_loss = -torch.mean(logprobs*advantages + epsilon*entropy)
    critic_loss = F.mse_loss(target_values.detach(),values)
    total_loss = actor_loss + critic_loss
    return total_loss, actor_loss, critic_loss

def A3C_train(rank,args,shared_agent):
    #env = make_env(True,123+process_number)
    env = make_env(True,args.seed+rank)
    optimizer = optim.Adam(shared_agent.parameters(),lr=args.lr)
    
    start = time.time()
    A3C_record = []
    reward_record, TDloss_record, ACloss_record, CRloss_record = [], [], [], []
    for ep in range(max_episode):
        print(f'에피소드: {ep} - 프로세스: {process_number}')
        done = False
        state = env.reset()
        cnt = 0
        total_reward = 0
        total_episode_TD = 0
        total_episode_acloss = 0
        total_episode_crloss = 0
        
        while True:
            torch_state = torch.Tensor(state).to(device)
            torch_state = torch.unsqueeze(torch_state,0)
            action = shared_agent.sample_actions(torch_state)
            next_state,reward,done,_ = env.step(action)
            total_reward += reward
            
            transition = (state,action,np.array([reward]),next_state,done)
            loss,actor_loss,critic_loss = A2C_loss(transition,shared_agent,env,gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_episode_TD += loss.item()
            total_episode_acloss += actor_loss.item()
            total_episode_crloss += critic_loss.item()
            if done:
                ep +=1 
                TDloss_record.append(total_episode_TD/cnt)
                ACloss_record.append(total_episode_acloss/cnt)
                CRloss_record.append(total_episode_crloss/cnt)
                reward_record.append(total_reward)
                if total_reward == 20:
                    best_agent = copy.deepcopy(shared_agent)
                break
            
            # 업데이트
            state = next_state
            cnt += 1
        finish = time.time()-start
        A3C_record.append([finish,total_reward]) 
        #if ep % 10 == 0:
        #    print(f'{ep}번째 에피소드 결과 - 프로세스: {process_number}')
        #    print(f'최근 10 에피소드 보상평균 = {np.mean(reward_record[-10:])}')
        #    print(f'최근 10 에피소드 A2C오차 = {np.mean(TDloss_record[-10:])}')
        #    
        #if np.mean(reward_record[-10:]) >= 20:
        #    best_agent = copy.deepcopy(shared_agent)
        #    print(f"충분한 보상: {np.mean(reward_record[-10:])}")
        #    print(f"학습종료 - 프로세스: {process_number}")
        #    break
        if ep % 2 == 0:
            print(f'{ep}번째 에피소드 결과 - 프로세스: {process_number}')
            print(f'최근 2 에피소드 보상평균 = {np.mean(reward_record[-2:])}')
            print(f'최근 2 에피소드 A2C오차 = {np.mean(TDloss_record[-2:])}')
            
        if np.mean(reward_record[-2:]) >= 20:
            best_agent = copy.deepcopy(shared_agent)
            print(f"충분한 보상: {np.mean(reward_record[-2:])}")
            print(f"학습종료 - 프로세스: {process_number}")
            np.save('./Single_agent_timeVsreward.npy',np.array(A3C_record))
            break 
    return best_agent

# %%
learning_rate = 1e-04
step_size=1e+05
gamma=0.99
max_episode= 10000000

# %%
env = make_env(True,123)
env.reset()
n_actions = env.action_space.n
state_shape = env.observation_space.shape

train_agent = A3C_Agent(n_actions).to(device)
train_agent.share_memory()
optimizer = optim.Adam(train_agent.parameters(),lr=learning_rate)
#optimizer = my_optim.SharedAdam(train_agent.parameters(), lr=learning_rate)
#optimizer.share_memory()
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)

best_agent = A3C_train(train_agent,optimizer,0)





# %%
#mp.set_start_method('spawn')
#processes = []
#num_processes = 4 #  일반적 4코어
#for rank in range(num_processes):
#    #p = mp.Process(target=A3C_train, args=(train_agent,env,rank))
#    p = mp.Process(target=A3C_train, args=(train_agent,optimizer,rank))
#    p.start()
#    processes.append(p)
#for p in processes:
#    p.join()
#    
#best_agent = copy.deepcopy(train_agent)
#
## %%
#import gym.wrappers
#
#def record(state,agent,env,vid):
#    reward = 0
#
#    while True:
#        vid.capture_frame()
#        torch_state = torch.Tensor(state).to(device)
#        torch_state = torch.unsqueeze(torch_state,0)
#        action = agent.sample_actions(torch_state)
#        state,r,done,_ = env.step(action) 
#        
#        reward += r
#        #print(reward,done)
#        if done:
#            break
#    vid.close()
#    return reward
#
#env = make_env(clip_rewards=True,seed=12)
#vid = gym.wrappers.monitoring.video_recorder.VideoRecorder(env,path='./videos/Pong/A3CPong_best.mp4')
#vid.render_mode="rgb_array"
#state = env.reset()
##rewards = record(state,best_agent,env,vid)
#rewards = record(state,train_agent,env,vid)
#print(rewards)
