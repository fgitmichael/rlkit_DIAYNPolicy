from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from rlkit.torch.distributions import TanhNormal

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class PPOTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            vf,

            epsilon=0.05,
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

        self.epsilon = epsilon
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        obs = batch['observations']
        old_log_pi = batch['log_prob']
        advantage = batch['advantage']
        returns = batch['returns']
        actions = batch['actions']

        """
        Policy Loss
        """
        _, policy_mean, policy_log_std, _, _, policy_std, _, _ = self.policy(obs)
        print(actions.min(), actions.max())
        new_log_pi = TanhNormal(policy_mean, policy_std).log_prob(actions)

        # Advantage Clip
        ratio = torch.exp(new_log_pi - old_log_pi)
        left = ratio * advantage
        right = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage

        policy_loss = (-1 * torch.min(left, right)).mean()

        """
        VF Loss
        """
        v_pred = self.vf(obs)
        v_target = returns
        vf_loss = self.vf_criterion(v_pred, v_target)

        """
        Update networks
        """
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
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
                'New Log Pis',
                ptu.get_numpy(new_log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Old Log Pis',
                ptu.get_numpy(old_log_pi),
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
