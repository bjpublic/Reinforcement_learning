import sys
sys.path.append('../')
import numpy as np
import copy
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt

from gym.spaces.box import Box
from PIL import Image
from IPython.display import clear_output
from tqdm import trange


from material.atari_util import *

# 학습용도
x_train = np.linspace(-1.5,1.5,5000)
y_train = x_train**3-1.5*x_train**2+0.7*x_train+0.01*np.cos(x_train)+np.random.uniform(size=5000)
x_min,x_max = x_train.min(), x_train.max()
y_min,y_max = y_train.min(), y_train.max()
#plt.plot(x_train,y_train,'o',markersize=1,label='Data')
#plt.grid()
#plt.legend()
#plt.show()

from torch.utils.data import Dataset, DataLoader

def preprocess(x,y):
    x = (x-x_min)/(x_max-x_min)
    y = (y-y_min)/(y_max-y_min)
    return x,y

def postprocess(x,y):
    x = x*(x_max-x_min)+x_min
    y = y*(y_max-y_min)+y_min
    return x,y

class MyDataset(Dataset):
    def __init__(self,x,y):
        x,y = preprocess(x,y)
        self.x = x.reshape(-1,1)
        self.y = y.reshape(-1,1)
    def __getitem__(self,idx):
        return torch.FloatTensor(self.x[idx]), torch.FloatTensor(self.y[idx])
    def __len__(self):
        return len(self.x)

train_dataset = MyDataset(x_train,y_train)
train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True,num_workers=4)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.lin1 = nn.Linear(1,64)
        self.lin2 = nn.Linear(64,64)
        self.lin3 = nn.Linear(64,64)
        self.lin4 = nn.Linear(64,1)
    def forward(self,x):
        x = F.elu(self.lin1(x))
        x = F.elu(self.lin2(x))
        x = F.elu(self.lin3(x))
        x = self.lin4(x)
        return x
    
#model = MyModel()
#optimizer = optim.Adam(model.parameters(),lr=1e-04)
#scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1000,gamma=0.7)
#
#best_loss = np.inf
#for ep in range(500):
#    train_loss = 0
#    for x,y in train_dataloader:
#        y_infer = model(x)
#        loss = torch.mean((y-y_infer)**2)
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#        train_loss += loss.item()
#    train_loss/=len(train_dataloader)
#    if train_loss < best_loss:
#        best_loss = train_loss
#        best_model = copy.deepcopy(model)
#    if ep % 50 == 0:
#        print(f'   >  학습상황: {ep/500*100}%')
#        print(f'      >> 훈련오차: {train_loss}, 최저오차: {best_loss}')
#print(f'         >>> 학습종료')

# 플롯 - 테스트 데이터 셋 확인
#x_test = np.linspace(-1.5,1.5,100)
#y_test = x_test**3-1.5*x_test**2+0.7*x_test+0.01*np.cos(x_test)
#x_test,y_test = preprocess(x_test,y_test)
#x_test = torch.Tensor(x_test).view(-1,1)
#y_test = torch.Tensor(y_test).view(-1,1)
#with torch.no_grad():
#    y_infer = model(x_test)
#    test_loss = torch.mean((y_infer-y_test)**2)
#print(f'   >  시험오차: {test_loss.item()}')
#x_test,y_infer = postprocess(x_test,y_infer)
#plt.plot(x_train,y_train,'o',markersize=1,label='Data')
#plt.plot(x_test.detach().cpu().numpy(),y_infer.detach().cpu().numpy(),label='Inference')
#plt.grid()
#plt.legend()
#plt.show()


import os

# Multiprocessing
def train(rank,share_model,train_dataloader):
    print(f'시작 프로세스: {rank}')
    #optimizer = optim.Adam(share_model.parameters(),lr=1e-04)
    #scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1000,gamma=0.7)
    pid = os.getpid()
    best_loss = np.inf
    for ep in range(500):
        train_loss = 0
        for x,y in train_dataloader:
            y_infer = share_model(x)
            loss = torch.mean((y-y_infer)**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss/=len(train_dataloader)
        if train_loss < best_loss:
            best_loss = train_loss
            best_model = copy.deepcopy(share_model)
        if ep % 50 == 0:
            print(f'   >  학습상황: {ep/500*100}%')
            print(f'      >> 훈련오차: {train_loss}, 최저오차: {best_loss}')
    print(f'         >>> 학습종료')
    
def test(share_model):
    x_test = np.linspace(-1.5,1.5,100)
    y_test = x_test**3-1.5*x_test**2+0.7*x_test+0.01*np.cos(x_test)
    x_test,y_test = preprocess(x_test,y_test)
    x_test = torch.Tensor(x_test).view(-1,1)
    y_test = torch.Tensor(y_test).view(-1,1)
    with torch.no_grad():
        y_infer = share_model(x_test)
        test_loss = torch.mean((y_infer-y_test)**2)
    print(f'   >  시험오차: {test_loss.item()}')
    x_test,y_infer = postprocess(x_test,y_infer)
    plt.plot(x_train,y_train,'o',markersize=1,label='Data')
    plt.plot(x_test.detach().cpu().numpy(),y_infer.detach().cpu().numpy(),label='Inference')
    plt.grid()
    plt.legend()
    plt.show()

multi_model = MyModel()
multi_model.share_memory()
optimizer = optim.Adam(multi_model.parameters(),lr=1e-04)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1000,gamma=0.7)

num_processes = 4
processes = []
for rank in range(num_processes):
    p = mp.Process(target=train,args=(rank,multi_model,train_dataloader))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
    
test(multi_model)
