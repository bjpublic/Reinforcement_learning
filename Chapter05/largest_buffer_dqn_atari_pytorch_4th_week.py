import sys, os
sys.path.append('..')

import random
import numpy as np
import torch
import torch.nn as nn

from material.atari_util import *
from material.replay_buffer import ReplayBuffer
from material.atari_wrapper import *
from material.framebuffer import FrameBuffer

import gym
import numpy as np
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description='Randomseed Argparse')
parser.add_argument('--seed',type=int,default=123)
parser.add_argument('--device',type=int,default=0)
args = parser.parse_args()

from gym.core import ObservationWrapper
from gym.spaces import Box

from PIL import Image # Your code

ENV_NAME = "BreakoutNoFrameskip-v4"

env = gym.make(ENV_NAME)
env.reset()

n_cols = 5
n_rows = 2
fig = plt.figure(figsize=(16, 9))


class PreprocessAtariObs(ObservationWrapper):
    def __init__(self, env):
        ObservationWrapper.__init__(self, env)

        self.img_size = (1, 64, 64)
        self.observation_space = Box(0.0, 1.0, self.img_size,dtype=np.float32)

    def _to_gray_scale(self, rgb, channel_weights=[0.8, 0.1, 0.1]):
        dummy = 0
        for idx,channel_weight in enumerate(channel_weights):
            dummy += channel_weight*(rgb[:,:,idx])
        return np.expand_dims(dummy,axis=-1)
    
    def observation(self, img):
        img = Image.fromarray(np.uint8(img),'RGB')
        img = img.resize((64,64))
        img = np.array(img)
        img = self._to_gray_scale(img)/255.
        #return np.array(img,dtype="float32").transpose((2,0,1))
        return np.array(img).transpose((2,0,1))

# %%
env = gym.make(ENV_NAME)  # create raw env
env = PreprocessAtariObs(env)
observation_shape = env.observation_space.shape
n_actions = env.action_space.n
env.reset()
obs, _, _, _ = env.step(env.action_space.sample())

# test observation
#assert obs.ndim == 3, "observation must be [channel, h, w] even if there's just one channel"
#assert obs.shape == observation_shape
#assert obs.dtype == 'float32'
#assert len(np.unique(obs)) > 2, "your image must not be binary"
#assert 0 <= np.min(obs) and np.max(
#    obs) <= 1, "convert image pixels to [0,1] range"
#
#assert np.max(obs) >= 0.5, "It would be easier to see a brighter observation"
#assert np.mean(obs) >= 0.1, "It would be easier to see a brighter observation"
#
#print("Formal tests seem fine. Here's an example of what you'll get.")

n_cols = 5
n_rows = 2


def PrimaryAtariWrap(env, clip_rewards=True):
    assert 'NoFrameskip' in env.spec.id
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = PreprocessAtariObs(env)
    return env


def make_env(clip_rewards=True, seed=None):
    env = gym.make(ENV_NAME)  # create raw env
    if seed is not None:
        env.seed(seed)
    env = PrimaryAtariWrap(env, clip_rewards)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    return env

env = make_env()
env.reset()
n_actions = env.action_space.n
state_shape = env.observation_space.shape


device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')


def conv2d_size_out(size, kernel_size, stride):
    return (size - (kernel_size - 1) - 1) // stride  + 1

class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        self.seq = nn.Sequential(
            nn.Conv2d(4,16,3,stride=2), # [batch,16,32,32]
            nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2), # [batch,32,16,16]
            nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2), # [batch,64,7,7]
            nn.ReLU(),
            nn.Flatten(), #[batch, 3136]
            nn.Linear(3136,256), # [batch, 256]
            nn.ReLU(),
            nn.Linear(256,n_actions) # [batch, n_actions]
        )
        

    def forward(self, state_t):
        qvalues = self.seq(state_t)
        assert qvalues.requires_grad, "qvalues must be a torch tensor with grad"
        assert len(qvalues.shape) == 2 and qvalues.shape[0] == state_t.shape[0] and qvalues.shape[1] == n_actions
        return qvalues

    def get_qvalues(self, states):
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice([0, 1],
                                            batch_size,
                                            p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)

agent = DQNAgent(state_shape, n_actions, epsilon=0.5).to(device)

def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)
    return np.mean(rewards)


def play_and_record(initial_state, agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer. 
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    for n_step in range(n_steps):
        qvalues = agent.get_qvalues([s])
        action = agent.sample_actions(qvalues)[0]
        new_s, r, done, _ = env.step(action)
        exp_replay.add(s,action,r,new_s,done)
        
        s = new_s
        sum_rewards += r
        if done:
            s = env.reset()
            
    return sum_rewards, s

target_network = DQNAgent(agent.state_shape, agent.n_actions, epsilon=0.5).to(device)
target_network.load_state_dict(agent.state_dict())

def compute_td_loss(states, actions, rewards, next_states, is_done,
                    agent, target_network,
                    gamma=0.99,
                    check_shapes=False,
                    device=device):
    """ Compute td loss using torch operations only. Use the formulae above. """
    states = torch.tensor(states, device=device, dtype=torch.float)    # shape: [batch_size, *state_shape]

    actions = torch.tensor(actions, device=device, dtype=torch.long)    # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)  # shape: [batch_size]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done.astype('float32'),
        device=device,
        dtype=torch.float
    )  # shape: [batch_size]
    is_not_done = 1 - is_done

    predicted_qvalues = agent(states)
    predicted_next_qvalues = target_network(next_states)
    
    predicted_qvalues_for_actions = predicted_qvalues[range(
        len(actions)), actions]

    next_state_values = predicted_next_qvalues.max(dim=-1).values

    assert next_state_values.dim(
    ) == 1 and next_state_values.shape[0] == states.shape[0], "must predict one value per state"

    target_qvalues_for_actions = rewards+is_not_done*gamma*next_state_values

    loss = torch.mean((predicted_qvalues_for_actions -
                       target_qvalues_for_actions.detach()) ** 2)

    if check_shapes:
        assert predicted_next_qvalues.data.dim(
        ) == 2, "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim(
        ) == 1, "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim(
        ) == 1, "there's something wrong with target q-values, they must be a vector"

    return loss


#obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(
#    10)
#
#loss = compute_td_loss(obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch,
#                       agent, target_network,
#                       gamma=0.99, check_shapes=True)
#loss.backward()


# %%
"""
## Main loop


It's time to put everything together and see if it learns anything.
"""

# %%
#seed = 123 #<YOUR CODE: your favourite random seed>
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# %%
env = make_env(seed)
state_shape = env.observation_space.shape
n_actions = env.action_space.n
state = env.reset()

agent = DQNAgent(state_shape, n_actions, epsilon=1).to(device)
target_network = DQNAgent(state_shape, n_actions).to(device)
target_network.load_state_dict(agent.state_dict())

# %%
"""
Buffer of size $10^4$ fits into 5 Gb RAM.

Larger sizes ($10^5$ and $10^6$ are common) can be used. It can improve the learning, but $10^4$ is quiet enough. $10^2$ will probably fail learning.
"""

# %%
exp_replay = ReplayBuffer(10**6)
for i in range(100):
    #if not utils.is_enough_ram(min_available_gb=0.1):
    if not is_enough_ram(min_available_gb=0.1):
        print("""
            Less than 100 Mb RAM available. 
            Make sure the buffer size in not too huge.
            Also check, maybe other processes consume RAM heavily.
            """
             )
        break
    play_and_record(state, agent, env, exp_replay, n_steps=10**2)
    if len(exp_replay) == 10**4:
        break

# %%
timesteps_per_epoch = 1
batch_size = 16
total_steps = 10 * 10**6
decay_steps = 10**6

opt = torch.optim.Adam(agent.parameters(), lr=1e-4)

init_epsilon = 1
final_epsilon = 0.1

loss_freq = 50
refresh_target_network_freq = 5000
eval_freq = 5000

max_grad_norm = 50

n_lives = 5

# %%
mean_rw_history = []
td_loss_history = []
grad_norm_history = []
initial_state_v_history = []
step = 0

# %%
RW_record = 0
state = env.reset()
for step in range(step, total_steps + 1):
    #if not utils.is_enough_ram():
    if not is_enough_ram():
        print('less that 100 Mb RAM available, freezing')
        print('make sure everything is ok and make KeyboardInterrupt to continue')
        try:
            while True:
                pass
        except KeyboardInterrupt:
            pass

    agent.epsilon = linear_decay(init_epsilon, final_epsilon, step, decay_steps)

    # play
    _, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)

    # train
    obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(batch_size)
    
    loss = compute_td_loss(obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch,
                       agent, target_network,
                       gamma=0.99, check_shapes=False)

    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    opt.step()
    opt.zero_grad()

    if step % loss_freq == 0:
        td_loss_history.append(loss.data.cpu().item())
        grad_norm_history.append(grad_norm)

    if step % refresh_target_network_freq == 0:
        # Load agent weights into target_network
        target_network.load_state_dict(agent.state_dict())

    if step % eval_freq == 0:
        Mean_rw_hist=evaluate(
            make_env(clip_rewards=True, seed=step), agent, n_games=3 * n_lives, greedy=True)
        
        initial_state_q_values = agent.get_qvalues(
            [make_env(seed=step).reset()]
        )
        print(f'------------------------------')
        print(f'Epoch: {step}')
        print(f'Mean RW_history: {Mean_rw_hist}')
        print(f'Initial Q value: {np.max(initial_state_q_values)}')
        print('')

    if Mean_rw_hist > RW_record and (step+1) % 1e+04 == 0:
        RW_record = Mean_rw_hist
        print('Check point save')
        torch.save(agent,f'./ckpt/Randomseed_{seed}_ckpt_{step}_Atari_model.pth')

    if Mean_rw_hist >= 100:
        print('---------------------------------')
        print('Now, agent palys well as much as Human level!')
        print('Train process finished!')
        torch.save(agent,f'./ckpt/Randomseed_{seed}_ckpt_{step}_Atari_model.pth')
        break

# %%
final_score = evaluate(
  make_env(clip_rewards=False, seed=9),
    agent, n_games=30, greedy=True, t_max=10 * 1000
) * n_lives
print('final score:', final_score)
assert final_score >= 10, 'not as cool as DQN can'
print('Cool!')
