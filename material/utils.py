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

#def discounted_reward(rewards,next_values,dones,gamma=0.99):
#    target_value = []
#    discounted_reward = next_values[-1]*dones[-1]
#    for reward, next_value, done in zip(reversed(rewards),reversed(next_values),reversed(dones)):
#        discounted_reward = reward + next_value*done*discounted_reward * gamma
#        target_value.insert(0, discounted_reward)
#    return target_value    
    
def FIFO(element, array, length=400):
    if len(array) < length:
        array.append(element)
    else:
        array = array[1:]
        array.append(element)
    return array


#def moving_average(x, span=100):
#    return pd.DataFrame({'x': np.asarray(x)}).x.ewm(span=span).mean().values