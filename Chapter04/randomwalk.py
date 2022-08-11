# Origin code reference: https://towardsdatascience.com/reinfoocement-learninr-td-lambda-introduction

import numpy as np
import matplotlib.pyplot as plt

NUM_STATES=19
START = 9
END_0 = 0
END_1 = 20

class ValueFunction:
    def __init__(self,alpha=0.1):
        self.weights = np.zeros(NUM_STATES+2)
        self.alpha = alpha

    def value(self,state):
        v = self.weights[state]
        return v

    def learn(self,state,delta):
        self.weights[state] += self.alpha*delta

class RandomWalk:
    def __init__(self,start=START,end=False,lmbda=0.4,debug=False):
        self.actions = ["left","right"]
        self.state = start # current state
        self.end = end
        self.lmbda = lmbda
        self.reward = 0
        self.debug = debug
        self.rate_truncate = 1e-03

    def chooseAction(self):
        action = np.random.choice(self.actions)
        return action

    def takeAction(self,action):
        new_state = self.state
        if not self.end:
            if action == "left":
                new_state = self.state-1
            else:
                new_state = self.state+1

            if new_state in [END_0, END_1]:
                self.end = True
        self.state = new_state
        return self.state

    def giveReward(self,state):
        if state == END_0:
            return -1
        elif self == END_1:
            return 1
        else:
            return 0

    def reset(self):
        self.state = START
        self.end = False
        self.states = []

    def gt2tn(self,valueFunc,start,end):
        # only the last reward is non-zero
        reward = self.reward if end == len(self.states) -1 else 0
        state = self.states[end]
        res = reward + valueFunc.value(state)
        return res

    def play(self,valueFunc,rounds=100):
        for _ in range(rounds):
            self.reset()
            action=self.chooseAction()

            self.states = [self.state]
            while not self.end:
                state = self.takeAction(action) # next state
                self.reward = self.giveReward(state) # next state_reward

                self.states.append(state)
                action = self.chooseAction()
            if self.debug:
                print(f'Total states {len(self.states)} end at {self.state}: reward_{self.reward}')

            # end of game, do forward update
            T = len(self.states)-1
            for t in range(T):
                # start from time T
                state = self.states[t]
                gtlambda=0
                for n in range(1,T-t):
                    # compute G_t:t+n
                    gttn = self.gt2tn(valueFunc,t,t+n)
                    lambda_power = self.lmbda**(n-1)
                    gtlambda += lambda_power*gttn
                    if lambda_power < self.rate_truncate:
                        break
                
                gtlambda *= 1-self.lmbda
                if lambda_power >= self.rate_truncate:
                    gtlambda += lambda_power*self.reward
                delta = gtlambda - valueFunc.value(state)

class ValueFunctionTD:
    def __init__(self,alpha=0.1,gamma=0.9,lmbda=0.8):
        self.weights = np.zeros(NUM_STATES+2)
        self.z = np.zeros(NUM_STATES+2)
        self.alpha = alpha
        self.gamma = gamma
        self.lmbda = lmbda

    def value(self,state):
        v = self.weights[state]
        return v

    def updateZ(self,state):
        dev = 1
        self.z *= self.gamma * self.lmbda
        self.z[state] += dev

    def learn(self,state,nxtState,reward):
        delta = reward + self.gamma*self.value(nxtState)-self.value(state)
        delta *= self.alpha
        self.weights += delta * self.z

class RWTD:
    def __init__(self,start = START,end=False, debug=False):
        self.actions = ["left","right"]
        self.state = start # current state
        self.end = end
        self.reward = 0
        self.debug = debug

    def chooseAction(self):
        action = np.random.choice(self.actions)
        return action

    def takeAction(self,action):
        new_state = self.state
        if not self.end:
            if action == "left":
                new_state = self.state-1
            else:
                new_state = self.state+1

            if new_state in [END_0, END_1]:
                self.end = True
        return new_state

    def giveReward(self,state):
        if state == END_0:
            return -1
        elif state == END_1:
            return 1
        else:
            return 0

    def reset(self):
        self.state = START
        self.end = False
        self.states = []

    def play(self,valueFunc,rounds=100):
        for _ in range(rounds):
            self.reset()
            action = self.chooseAction()
            while not self.end:
                nxtState = self.takeAction(action) # next state
                self.reward = self.giveReward(nxtState) # next state-reward
                
                valueFunc.updateZ(self.state)
                valueFunc.learn(self.state,nxtState,self.reward)

                self.state = nxtState
                action = self.chooseAction()
            if self.debug:
                print(f'End at {self.state}: reward_{self.reward}')

if __name__ == "__main__":
    actual_state_values = np.arange(-20,22,2)/20.
    actual_state_values[0] = actual_state_values[-1] = 0
    actual_state_values

    alphas = np.linspace(0,0.8,6)
    lambdas = np.linspace(0,0.99,5)
    rounds=50

    # offline-lambda 
    plt.figure(figsize=[10,6])
    for lamb in lambdas:
        alpha_erros = []
        for alpha in alphas:
            valueFunc = ValueFunction(alpha=alpha)
            rw = RandomWalk(debug=False,lmbda=lamb)
            rw.play(valueFunc,rounds=rounds)
            rmse = np.sqrt(np.mean(np.power(valueFunc.weights-actual_state_values,2)))
            alpha_erros.append(rmse)
        plt.plot(alphas,alpha_erros,label=f"lambda={lamb}")

    plt.xlabel("alpha",size=14)
    plt.ylabel("RMS error",size=14)
    plt.legend()
    plt.show()

    # TD lambda
    alphas = np.linspace(0,0.8,6)
    lambdas = np.linspace(0,0.95,5)
    rounds=50
    plt.figure(figsize=[10,6])
    for lamb in lambdas:
        alpha_erros=[]
        for alpha in alphas:
            valueFunc=ValueFunctionTD(alpha=alpha,lmbda=lamb)
            rw=RWTD(debug=False)
            rw.play(valueFunc,rounds=rounds)
            rmse=np.sqrt(np.mean(np.power(valueFunc.weights-actual_state_values,2)))
            print(f"lambda {lamb} alpha {alpha} rmse {rmse}")
            alpha_erros.append(rmse)
        plt.plot(alphas,alpha_erros,label=f"lambda={lamb}")
    plt.xlabel("alpha",size=14)
    plt.ylabel("RMS error",size=14)
    plt.legend()
    plt.show()
                    


        
