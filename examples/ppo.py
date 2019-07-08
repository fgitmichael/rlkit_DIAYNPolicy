from gym.envs.mujoco import HalfCheetahEnv
from gym.envs.classic_control import PendulumEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.ppo.ppo_env_replay_buffer import PPOEnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector.path_collector import MdpPathCollector
from rlkit.torch.ppo.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.ppo.ppo import PPOTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.ppo.ppo_torch_batch_rl_algorithm import PPOTorchBatchRLAlgorithm

import torch
def experiment(variant):
    torch.autograd.set_detect_anomaly(True)
#    expl_env = NormalizedBoxEnv(HalfCheetahEnv())
#    eval_env = NormalizedBoxEnv(HalfCheetahEnv())
    expl_env = NormalizedBoxEnv(PendulumEnv())
    eval_env = NormalizedBoxEnv(PendulumEnv())
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    vf = FlattenMlp(
        input_size=obs_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_step_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_step_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = PPOEnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = PPOTrainer(
        env=eval_env,
        policy=policy,
        vf=vf,
        **variant['trainer_kwargs']
    )
    algorithm = PPOTorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_step_collector,
        evaluation_data_collector=eval_step_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="PPO",
        version="normal",
        layer_size=256,
        replay_buffer_size=128,
        algorithm_kwargs=dict(
            num_iter=200,
            num_eval_steps_per_epoch=200,
            num_trains_per_train_loop=int(2048/64*3),
            num_expl_steps_per_train_loop=2048,
            min_num_steps_before_training=100,
            max_path_length=200,
            minibatch_size=64,
        ),
        trainer_kwargs=dict(
            epsilon=0.05,
            gae_lambda=0.95,
            discount=0.99,
            reward_scale=1.0,
            policy_lr=3E-4,
            vf_lr=3E-4,
        ),
    )
    setup_logger('name-of-experiment', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
