from env import Maze
from test15_Sarsa_1 import QLearningTable


def update():
    for epoch in range(100):

        observation = env.reset()

        action = RL.choose_action(str(observation))

        while True:
            env.render()

            observation_, r, done = env.step(action)

            action_ = RL.choose_action(str(observation_))

            RL.learn(str(observation), action, r, str(observation_), action_)

            observation = observation_
            action = action_

            if done:
                break

    env.destroy()


env = Maze()
RL = QLearningTable(actions=list(range(env.n_actions)))
env.after(100, update)
env.mainloop()
