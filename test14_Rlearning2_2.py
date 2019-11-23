from env import Maze
from test14_Rlearning2_1 import QLearningTable


def update():
    for epoch in range(100):

        observation = env.reset()

        while True:
            env.render()

            action = RL.choose_action(str(observation))

            observation_, r, done = env.step(action)

            RL.learn(str(observation), action, r, str(observation_))

            observation = observation_

            if done:
                break

    env.destroy()


env = Maze()
RL = QLearningTable(actions=list(range(env.n_actions)))
env.after(100, update)
env.mainloop()
