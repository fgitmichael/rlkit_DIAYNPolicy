from gym import spaces
import numpy as np

class SanityEnv:
    def __init__(self):
        self.goal = 10
        self.pos = 0

        self.action_space = spaces.Box(-1, 1, (1,))
        self.observation_space = spaces.Box(0, 20, (1,))

    def reset(self):
        self.pos = 0

        return np.array([self.pos])

    def step(self, action):
        self.pos += action[0]
        self.pos = max(min(self.pos, 20), -20)
        r = int(abs(self.goal - self.pos) < 1)

        return np.array([self.pos]), r, False, {}
