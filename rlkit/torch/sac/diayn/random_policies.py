class SkillTanhGaussianPolicy(TanhGaussianPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            skill_dim=10,
            **kwargs
    ):
        super().__init__(
            hidden_sizes=hidden_sizes,
            obs_dim=obs_dim,
            action_dim=action_dim,
            std=std,
            init_w=init_w,
            **kwargs
        )
        self.skill_dim = skill_dim
        self.skill = 0

    def get_action(self, obs_np, deterministic=False):
        # generate (iters, skill_dim) matrix that stacks one-hot skill vectors
        # online reinforcement learning
        skill_vec = np.zeros(self.skill_dim)
        skill_vec[self.skill] += 1
        obs_np = np.concatenate((obs_np, skill_vec), axis=0)
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {"skill": skill_vec}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def skill_reset(self):
        self.skill = random.randint(0, self.skill_dim-1)

    def forward(
            self,
            obs,
