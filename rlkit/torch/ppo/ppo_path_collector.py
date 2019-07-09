from rlkit.samplers.data_collector.path_collector import MdpPathCollector
from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.core import torch_ify, np_ify
import numpy as np
import torch

class PPOMdpPathCollector (MdpPathCollector):
    def __init__(
            self,
            env,
            policy,

            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,

            calculate_advantages = False,
            vf = None,
            discount=0.99,
            gae_lambda=0.95
    ):
        self.calculate_advantages = calculate_advantages
        self.vf = vf
        self.discount = discount
        self.gae_lambda = gae_lambda
        super().__init__(
            env,
            policy,

            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None
        )

    """Generalized Advantage Estimator"""
    def add_advantages(self, path, path_len, flag):
        if flag:
            delta = torch_ify(path["rewards"]) + self.discount * self.vf(torch_ify(path["next_observations"])) - self.vf(torch_ify(path["observations"]))
            coef = torch.ones((path_len, path_len))
            for i in range(path_len):
                for j in range(i, path_len):
                    coef[i, j] *= (self.discount * self.gae_lambda) ** (j - i)
            advantages = np_ify(torch.matmul(coef, delta))
        else:
            advantages = np.zeros(path_len)
        return dict(
            observations=path["observations"],
            actions=path["actions"],
            rewards=path["rewards"],
            next_observations=path["next_observations"],
            terminals=path["terminals"],
            agent_infos=path["agent_infos"],
            env_infos=path["env_infos"],
            advantages=advantages
        )

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
            )
            path_len = len(path['actions'])

            # calculate advantages and add column to path
            path = self.add_advantages(path, path_len, self.calculate_advantages)

            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths
