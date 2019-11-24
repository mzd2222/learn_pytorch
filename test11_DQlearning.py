import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# DQN三种改进  1.double DQN   2.记忆重要性排序  3.输出层变为两层


global losses
losses = []

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.8  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 50  # target update frequency
MEMORY_CAPACITY = 1500
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(),
                              int) else env.action_space.sample().shape  # to confirm the shape


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 128)
        self.fc1.weight.detach().normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(128, 64)
        self.fc2.weight.detach().normal_(0, 0.1)  # initialization
        self.fc3 = nn.Linear(64, 32)
        self.fc3.weight.detach().normal_(0, 0.1)  # initialization
        self.out = nn.Linear(32, N_ACTIONS)
        self.out.weight.detach().normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


#########################################################
# env = gym.make('CartPole-v0')
#
# global dd
#
# dd = Net()
# dd.load_state_dict(torch.load('./model/ql.pkl'))
#
#
# def choose_action(x):
#     x = torch.unsqueeze(torch.FloatTensor(x), 0)
#     # input only one sample
#
#     actions_value = dd(x)
#     action = torch.max(actions_value, 1)[1].item()
#
#     return action
#
# step = 0
# for i_episode in range(20):
#     observation = env.reset()  # 重新开始游戏
#
#     while 1:
#         env.render()  # 刷新画面
#         step += 1
#         # print(observation)
#         # action = env.action_space.sample()
#         # print(action)
#         action = choose_action(observation)
#         observation, reward, done, info = env.step(action)
#         #print('observation :', observation, ',reward :', reward)
#         if done:
#             print("Episode finished after {} timesteps".format(step + 1))
#             step = 0
#             break
# print(env.action_space)  # 输入空间
# print(env.observation_space)  # 输出空间
# env.close()
#
#


# dueling
#
##############################################################
class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
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
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
        # print(transition)
        # print(transition.shape)
        # print(self.memory.shape)
        # exit(0)

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            # print('asd')
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        # print('b_s:', self.eval_net(b_s).size(), 'b_a', b_a.size())
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        a = loss.detach().numpy()
        losses.append(np.mean(a))


max_core = 0
dqn = DQN()
draw = []
print('\nCollecting experience...')
for i_episode in range(180):
    s = env.reset()
    # print(s)
    # exit(0)
    ep_r = 0
    while True:
        env.render()
        a = dqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)

        # print(s_)
        # exit(0)
        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_)

        ep_r += r  # 得分

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if EPSILON < 0.96:
                EPSILON = EPSILON * 1.00001
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2),
                      '| EPSILON: ', round(EPSILON, 2))
                draw.append([i_episode, round(ep_r, 2)])
                if ep_r > max_core:
                    max_core = ep_r
                    torch.save(dqn.eval_net.state_dict(), './model/ql.pkl')
        if done:
            break
        s = s_

draw = np.array(draw)
env.close()
plt.plot(losses)
plt.show()
