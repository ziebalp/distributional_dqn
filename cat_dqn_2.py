import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T



env = gym.make('CartPole-v0').unwrapped
state_shape = env.observation_space.shape
action_count = env.action_space.n

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','terminated'))

class LinearSchedule(object):
    def __init__(self, start, end, steps):
        self.steps = steps
        self.start = start
        self.end = end

    def value(self, t):
        fraction = min(float(t) / self.steps, 1.0)
        return self.start + fraction * (self.end - self.start)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

BATCH_SIZE = 32
GAMMA = 0.6
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

class DQNNet(nn.Module):
    def __init__(self):
        super(DQNNet, self).__init__()
        self.lin1 = torch.nn.Linear(4, 64)
        self.head = torch.nn.Linear(64, 9)
       

    def forward(self, x):
        x = F.relu(self.lin1(x))
        out = self.head(x)
        #print(x.size())
        #splits = out.view(x.size()[0],2,9).chunk(x.size()[0])
        #print(splits)
        #return torch.stack(list(map(lambda s: F.softmax(s[0]), splits)), 0)
        return F.softmax(out)

model0 = DQNNet()
model1 = DQNNet()
if use_cuda:
    model0.cuda()
    model1.cuda()

optimizer0 = optim.Adam(model0.parameters(),lr=0.00025)
optimizer1 = optim.Adam(model1.parameters(),lr=0.00025)

memory = ReplayMemory(10000)


steps_done = 0

class DQNAgent:

    def __init__(self, state_shape, action_count,):
        self.state_shape = state_shape
        self.action_count = action_count
        self.gamma = 0.6
        self.learning_rate = 0.00025
        self.momentum = 0.95

        # Distribution parameters
        self.vmin = -4
        self.vmax = 4
        self.n = 9
        self.dz = (self.vmax - self.vmin) / (self.n - 1)

        # Support atoms
        self.z = np.linspace(self.vmin, self.vmax, self.n, dtype=np.float32)

    def select_action(self, state,epsilon):
        sample = random.random()
        #print(epsilon)
        if sample > epsilon:
            mult = np.zeros((9,1))
            mult[:,0]=self.z
            #print(np.dot(model0(Variable(state, volatile=True).type(FloatTensor)).data.numpy(),self.z))
            #print(model1(Variable(state, volatile=True).type(FloatTensor)))
            val0 = np.dot(model0(Variable(state, volatile=True).type(FloatTensor)).data.numpy(),self.z)
            val1 = np.dot(model1(Variable(state, volatile=True).type(FloatTensor)).data.numpy(),self.z)
            action_returns = np.array([val0,val1])
            #print("returns ")
            #print(mult)
            #print(action_returns)
            #print(np.argmax(action_returns))
            return LongTensor([[int(np.argmax(action_returns))]])
        else:
            return LongTensor([[random.randrange(2)]])


    episode_durations = []

    def optimize_model(self):
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        next_states = Variable(torch.cat(batch.next_state))
        terminals=np.array(batch.terminated)

        # P[k, a, i] is the probability of atom z_i when action a is taken in the next state (for the kth sample)
        P0 = model0(next_states)
        P1 = model1(next_states)
        #print(torch.sum(P0[0]))
        #print(torch.sum(P1[0])) 
        # Q[k, a] is the value of action a (for the kth sample)
        #print(np.dot(P0.data.numpy(), self.z))
        Q = np.vstack((np.dot(P0.data.numpy(), self.z),np.dot(P1.data.numpy(), self.z))).T
        #print(Q)
        # A_[k] is the optimal action (for the kth sample)
        A_ = np.argmax(Q, axis=1)
        #print(A_)

        # Target vector
        M = np.zeros((BATCH_SIZE, self.action_count, self.n), dtype=np.float32)
        #print(reward_batch.data.numpy())
        # Compute projection onto the support (for terminal states, just reward)
        Tz = np.repeat(reward_batch.data.numpy().reshape(-1, 1), self.n, axis=1) + np.dot(self.gamma * (1.0 - terminals).reshape(-1, 1),
                                                                        self.z.reshape(1, -1))
        #print(self.gamma * (1.0 - terminals).reshape(-1, 1))
        
        # TODO: Verify correctnes
        # Clipping to endpoints like described in paper causes probabilities to disappear (when B = L = U).
        # To avoid this, I shift the end points to ensure that L and U are not both equal to B
        Tz = np.clip(Tz, self.vmin + 0.01, self.vmax - 0.01)
        

        B = (Tz - self.vmin) / self.dz
        L = np.floor(B).astype(np.int32)
        U = np.ceil(B).astype(np.int32)

        # Distribute probability
        for i in range(BATCH_SIZE):
            for j in range(self.n):
                if(A_[i]==0):
                    M[i, A_[i], L[i, j]] += P0[i, j].data[0] * (U[i, j] - B[i, j])
                    M[i, A_[i], U[i, j]] += P0[i, j].data[0] * (B[i, j] - L[i, j])
                else:
                    M[i, A_[i], L[i, j]] += P1[i, j].data[0] * (U[i, j] - B[i, j])
                    M[i, A_[i], U[i, j]] += P1[i, j].data[0] * (B[i, j] - L[i, j])

                #M[i, 1-A_[i], L[i, j]] = P[i, 1-A_[i], j].data[0]
                #M[i, 1-A_[i], U[i, j]] = P[i, 1-A_[i], j].data[0]

        #print("P:")
        #print(P[0])
        #print(M[0])
        #print("M:")
        #print(M[0])


        #print(A_)
        #print(action_batch)
        action_mask = LongTensor(A_).view(BATCH_SIZE,1,1).expand(BATCH_SIZE,1,9)
        #print(action_mask)
        
        q_probs0 = model0(state_batch)
        q_probs1 = model1(state_batch)
        #print(q_probs0[0])
        #print(q_probs1[0])
        qa_probs = [q_probs0[i] if action_batch[i].data[0] == 0 else q_probs1[i] for i in range(BATCH_SIZE)]
        #print(qa_probs[0])

        #criterion = nn.BCEWithLogitsLoss()
        #print(P.view(BATCH_SIZE,18)[0])
        #print(qa_probs[0])
        #print(M)
        matrix = Variable(Tensor(M)).gather(1, action_mask).squeeze()
        #print(matrix[0])
        #print(matrix[0])

        loss0=Variable(Tensor([0.0]))
        loss1=Variable(Tensor([0.0]))
        counter=0
        for i in range(BATCH_SIZE):
            #print(matrix[i] * torch.log(qa_probs[i]))
            #print(torch.sum(matrix[i] * torch.log(qa_probs[i])))
            #print(M[i][A_[i]])
            #print(matrix[i])
            if action_batch[i].data[0]==0:
                loss0 -= torch.sum(matrix[i] * torch.log(qa_probs[i]))
                counter+=1
            else:
                loss1 -= torch.sum(matrix[i] * torch.log(qa_probs[i]))

        #print(loss0)
        #print(loss1)
        if(counter>0):
            optimizer0.zero_grad()
            loss0.backward()
            optimizer0.step()
        if(counter<BATCH_SIZE):
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

num_episodes = 10000
running_reward = 10
agent = DQNAgent(state_shape,action_count)
epsilon_schedule = LinearSchedule(1, 0.01, 10000)

for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    previous_state=state
    state = env.reset()-previous_state
    for t in range(10000):
        # Select and perform an action
        action = agent.select_action(torch.from_numpy(state).float().unsqueeze(0),epsilon_schedule.value(i_episode))
        next_state, reward, done, _ = env.step(action[0,0])

        reward = Tensor([reward])

        #next_state = next_state-state
        #state = state - previous_state
        if done:
            #next_state = None
            memory.push(torch.from_numpy(state).float().unsqueeze(0), action, torch.from_numpy(next_state).float().unsqueeze(0), reward,1)    
        else:
            memory.push(torch.from_numpy(state).float().unsqueeze(0), action, torch.from_numpy(next_state).float().unsqueeze(0), reward,0)    

        # Store the transition in memory
        previous_state = state
        state=next_state

        # Perform one step of the optimization (on the target network)
        if done:
            break

    running_reward = running_reward * 0.99 + t * 0.01   
    agent.optimize_model()
    if i_episode % 10 == 0:
        
        print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
            i_episode, t, running_reward))
    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
        break