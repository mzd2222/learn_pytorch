import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

env = gym.make('Breakout-v0')

LR = 0.01
BATCH_SIZE = 32
MEMORYCOUNT = 1500

N_ACTIONS = env.action_space.n  # 4
N_STATES = env.observation_space.shape  # (210, 160, 3)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d()
        self.conv2 = nn.Conv2d()
        self.conv3 = nn.Conv2d()
        self.linear1 = nn.Linear()
        self.linear2 = nn.Linear()
        self.linear3 = nn.Linear()

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view()
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        return x


for i in range(5):
    s = env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        # print(action)

        # take action
        s_, r, done, info = env.step(action)

        if done:
            break

env.close()
