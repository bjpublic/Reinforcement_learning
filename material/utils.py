import pandas as pd
import numpy as np

def moving_average(x,span=100):
    return pd.DataFrame({'x':np.asarray(x)}).x.ewm(span=span).mean().values

def discounted_reward(rewards,gamma=0.99):
    target_value = []
    discounted_reward = 0
    for reward in reversed(rewards):
        discounted_reward = reward + discounted_reward * gamma
        target_value.insert(0, discounted_reward)
    return target_value
    
#def moving_average(x, span=100):
#    return pd.DataFrame({'x': np.asarray(x)}).x.ewm(span=span).mean().values