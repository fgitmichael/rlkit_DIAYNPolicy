from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
import numpy as np

class PPOEnvReplayBuffer(EnvReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None,

            discount,
            gae_lambda,
            epsilon
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon

        self._log_prob = np.zeros((max_replay_buffer_size, 1))
        self._advantage = np.zeros((max_replay_buffer_size, 1))

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            env=env,
            env_info_sizes=env_info_sizes
        )

    def calculate_advantage(self):
        # Generalized Advantage Estimator
        delta = self._rewards + self.discount * self.vf(self._next_obs) - self.vf(self._obs)
        coef = torch.ones(self._top)
        for i in range(1, self._top):
            coef[i:] *= self.discount * self.gae_lambda
        advantage = torch.sum(coef * delta)
        if advantage >= 0:
            e_advantage = advantage + self.epsilon
        else:
            e_advantage = advantage - self.epsilon
        self._advantage[:self._top] = e_advantage

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, agent_info,**kwargs):
        """
        Log Probability of action is stored in agent_info
        Empty Advantage is stored
        """
        self._log_prob[self._top] = agent_info["log_prob"]
        self._advantage[self._top] = [0]

        return super().add_sample(
            observation=observation, 
            action=action, 
            reward=reward, 
            terminal=terminal,
            next_observation=next_observation,
            **kwargs
        )

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            log_prob=self._log_prob[indices],
            advantage=self._advantage[indices]
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch
