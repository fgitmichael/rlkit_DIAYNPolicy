from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class PPOTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            vf,

            epsilon=0.1,
            gamma=0.01,
            gae_lambda=0.01,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            vf_lr=1e-3,
            optimizer_class=optim.Adam,

            plotter=None,
            render_eval_paths=False,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.vf = vf

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        print(batch.keys())
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        old_log_pi = batch['agent_infos']['log_pi']

        """
        Policy Loss
        """
        new_obs_actions, policy_mean, policy_log_std, new_log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )

        # Generalized Advantage Estimator
        delta = rewards + gamma * vf(next_obs) - vf(obs)
        coef = torch.ones(delta.size[0])
        for i in range(1, delta.size[0]):
            coef[i:] *= gae_lambda
        advantage = torch.sum(coef * delta)
        if advantage >= 0:
            e_advantage = advantage + epsilon
        else:
            e_advantage = advantage - epsilon

        policy_loss = (torch.min(
            torch.exp(old_log_pi-log_pi) * advantage,
            e_advantage
            )).mean()

        """
        VF Loss
        """
        v_pred = self.vf(obs)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        v_target = self.reward_scale * rewards \
            + (1. - terminals) * self.discount * self.vf(next_obs)
        vf_loss = self.vf_criterion(v_pred, v_target.detach())

        """
        Update networks
        """
        self.vf_optimizer.zero_grad()
        v_loss.backward()
        self.vf_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - v_new_actions).mean()

            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Target',
                ptu.get_numpy(v_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.vf
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            vf=self.vf
        )
