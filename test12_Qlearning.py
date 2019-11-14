import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

env = gym.make('Breakout-v0')

LR = 0.01
BATCH_SIZE = 32

MEMORYCOUNT = 2500
TARGET_REPLACE_ITER = 100

EPSILON = 0.8
GAMMA = 0.9

N_ACTIONS = env.action_space.n  # 4
N_STATES = 210 * 160 * 3  # (210, 160, 3)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)  #
        self.conv1.weight.detach().normal_(0, 0.1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.conv2.weight.detach().normal_(0, 0.1)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.conv3.weight.detach().normal_(0, 0.1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4.weight.detach().normal_(0, 0.1)
        self.linear1 = nn.Linear(256 * 13 * 10, 128)
        self.linear1.weight.detach().normal_(0, 0.1)
        self.linear2 = nn.Linear(128, 64)
        self.linear2.weight.detach().normal_(0, 0.1)
        self.linear3 = nn.Linear(64, N_ACTIONS)
        self.linear3.weight.detach().normal_(0, 0.1)

    def forward(self, input):
        x = input.view(input.size(0), 3, 210, 160)  # 1, 3, 210, 160
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)  # 1, 128, 26 ,20
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)  # 1, 256, 13 ,10

        x = x.view(x.size(0), -1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        return x


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = CNN(), CNN()
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = {}  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

        self.eval_net.cuda()
        self.target_net.cuda()
        self.loss_func.cuda()

    def choose_action(self, x1):
        x = torch.unsqueeze(torch.FloatTensor(x1), 0)

        x = x.cuda()

        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            actions_value = self.eval_net(x)
            action = torch.max(actions_value, 1)[1].item()
            # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:  # random
            action = np.random.randint(0, N_ACTIONS)
            # action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.array([np.array(s, dtype=np.float), a, r, np.array(s_, dtype=np.float)])
        # replace the old memory with new memory
        index = self.memory_counter % MEMORYCOUNT
        self.memory[index] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            # print('asd')
        self.learn_step_counter += 1

        # sample batch transitions
        b_memory = []
        sample_index = np.random.choice(len(self.memory.keys()), BATCH_SIZE)
        # print(self.memory.keys())
        for i in sample_index:
            b_memory.append(self.memory[i])
        b_memory = np.array(b_memory)

        b_s = torch.FloatTensor(np.stack(b_memory[:, 0]))
        b_a = torch.LongTensor(np.stack(b_memory[:, 1]).astype(int)).view(BATCH_SIZE, -1)
        b_r = torch.FloatTensor(np.stack(b_memory[:, 2])).view(BATCH_SIZE, -1)
        b_s_ = torch.FloatTensor(np.stack(b_memory[:, 3]))

        b_s = b_s.cuda()
        b_a = b_a.cuda()
        b_r = b_r.cuda()
        b_s_ = b_s_.cuda()

        # q_eval w.r.t the action in experience
        # print('b_s:', self.eval_net(b_s).size(), 'b_a', b_a.size())
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        # print('q_eval:', q_eval.size(), ' |q_target: ', q_target.size())
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

max_core = 0

for i in range(25):
    s = env.reset()
    ep_r = 0
    while True:
        env.render()

        action = dqn.choose_action(s)
        # print(action)

        # take action
        s_, r, done, info = env.step(action)

        dqn.store_transition(s, action, r, s_)
        # s_ = np.array(s_) / np.max(s_) - 0.001  # 归一化
        # s_t = torch.unsqueeze(torch.tensor(s_, dtype=torch.float32), dim=0)
        ep_r += r

        if dqn.memory_counter > MEMORYCOUNT:
            dqn.learn()

            if EPSILON < 0.90:
                EPSILON = EPSILON * 1.00001
            if done:
                print(EPSILON)
                print('Ep: ', i,
                      '| Ep_r: ', round(ep_r, 2),
                      '| EPSILON: ', round(EPSILON, 2))
                # draw.append([i, round(ep_r, 2)])
                if ep_r > max_core:
                    max_core = ep_r
                    torch.save(dqn.eval_net.state_dict(), './model/ql2.pkl')
        if done:
            break

        s = s_

env.close()
