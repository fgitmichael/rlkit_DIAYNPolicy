import gym
from gym.envs.mujoco import HalfCheetahEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.sac.diayn.diayn_env_replay_buffer import DIAYNEnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.diayn.d_diayn_path_collector import DirichletDIAYNMdpPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.torch.sac.diayn.policies import DirichletSkillTanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.diayn.d_diayn import DirichletDIAYNTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.diayn.d_diayn_torch_online_rl_algorithm import DirichletDIAYNTorchOnlineRLAlgorithm


def experiment(variant):
    # expl_env = NormalizedBoxEnv(HalfCheetahEnv())
    # eval_env = NormalizedBoxEnv(HalfCheetahEnv())
    expl_env = NormalizedBoxEnv(gym.make("BipedalWalker-v2"))
    eval_env = NormalizedBoxEnv(gym.make("BipedalWalker-v2"))
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    skill_dim = 10

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    df = FlattenMlp(
        input_size=obs_dim,
        output_size=skill_dim,
        hidden_sizes=[M, M],
    )
    policy = DirichletSkillTanhGaussianPolicy(
        obs_dim=obs_dim + skill_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        skill_dim=skill_dim
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = DirichletDIAYNMdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_step_collector = MdpStepCollector(
        expl_env,
        policy,
    )
    replay_buffer = DIAYNEnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        skill_dim,
    )
    trainer = DirichletDIAYNTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        df=df,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = DirichletDIAYNTorchOnlineRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_step_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="Dirichlet_DIAYN",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger('dirichlet_diayn_10_bipedalWalker', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
