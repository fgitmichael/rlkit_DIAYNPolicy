import gym
import argparse
#from gym.envs.mujoco import HalfCheetahEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.h_diayn.manager_ppo_env_replay_buffer import ManagerPPOEnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.h_diayn.manager_ppo_path_collector import ManagerPPOMdpPathCollector
from rlkit.torch.ppo.policies import DiscretePolicy, MakeDeterministic
from rlkit.torch.ppo.discrete_ppo import DiscretePPOTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.ppo.ppo_torch_batch_rl_algorithm import PPOTorchBatchRLAlgorithm

from sanity import SanityEnv

import torch

def experiment(variant):
    torch.autograd.set_detect_anomaly(True)
    # expl_env = NormalizedBoxEnv(HalfCheetahEnv())
    # eval_env = NormalizedBoxEnv(HalfCheetahEnv())
    expl_env = NormalizedBoxEnv(gym.make(str(args.env)))
    eval_env = NormalizedBoxEnv(gym.make(str(args.env)))
    obs_dim = expl_env.observation_space.low.size
    worker = torch.load(str(args.worker))['trainer/policy']
    skill_dim = worker.skill_dim

    M = variant['layer_size']
    vf = FlattenMlp(
        input_size=obs_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = DiscretePolicy(
        obs_dim=obs_dim,
        action_dim=skill_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_step_collector = ManagerPPOMdpPathCollector(
        eval_env,
        eval_policy,
        worker,
        calculate_advantages=False
    )
    expl_step_collector = ManagerPPOMdpPathCollector(
        expl_env,
        policy,
        worker,
        calculate_advantages=True,
        vf=vf,
        gae_lambda=0.97,
        discount=0.995,
    )
    replay_buffer = ManagerPPOEnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        skill_dim=skill_dim
    )
    trainer = DiscretePPOTrainer(
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
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str,
                        help='environment')
    parser.add_argument('worker', type=str,
                        help='saved params for worker policy')
    parser.add_argument('--skill_dim', type=int, default=10,
                        help='skill dimension')
    args = parser.parse_args()

    # noinspection PyTypeChecker
    T = 2048
    max_ep_len = 1000
    epochs = 10
    minibatch_size = 64

    variant = dict(
        algorithm="PPO_DIAYN",
        version="normal",
        layer_size=64,
        replay_buffer_size=T,
        algorithm_kwargs=dict(
            num_iter=int(1e6 // T),
            num_eval_steps_per_epoch=max_ep_len,
            num_trains_per_train_loop=T // minibatch_size * epochs,
            num_expl_steps_per_train_loop=T,
            min_num_steps_before_training=0,
            max_path_length=max_ep_len,
            minibatch_size=minibatch_size,
        ),
        trainer_kwargs=dict(
            epsilon=0.2,
            reward_scale=1.0,
            lr=3e-4,
        ),
    )
    setup_logger('PPODIAYN_' + args.env, variant=variant)
    #ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
