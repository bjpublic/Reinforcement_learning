import numpy as np
import matplotlib.pyplot as plt

class Random_walk():
    def __init__(self,num_states=5,gamma=0.9,alpha=0.5):
        self.num_states = num_states
        self.gamma = gamma
        self.alpha = alpha

        self.actions = ['left','right']
        self.position = self.num_states//2 + 1
        self.agent_value = 0
    
    def reset():
       print(f'Agent have reached end position. {self.position}')
       self.position = self.num_states//2+1
       self.agent_value = 0

    def is_done():
        if self.position < 1 or self.position > self.num_states:
            return True
        else:
            return False

    def random_action():
        action = np.random.choice(self.actions)
        if action == 'left':
            self.position -= 1
        else:
            self.position += 1
    
    def get_reward():
        reward = 0
        if self.position == 0:
            reward = -1
        if self.position == self.num_states+1:
            reward = +=1
        return reward

    def play(reset=False):
        if reset:
            self.reset()
            G = 0
        while not self.is_done():
            self.random_action()
            G += self.get_reward()+gamma*G 
        self.agent_value += self.alpha*(G-self.agent_value)
        return self.agent_value

agent = Random_walk()


ns = [1000]
for n in ns:
    value = agent.play()
    
    plt.plot(range(num_states),values,'-*',label=f'Iteration: {n}')
plt.grid()
plt.legend()
plt.show()
