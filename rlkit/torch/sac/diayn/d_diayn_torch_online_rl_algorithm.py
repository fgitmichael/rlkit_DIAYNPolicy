import gtimer as gt
from rlkit.torch.sac.diayn.diayn_torch_online_rl_algorithm import DIAYNTorchOnlineRLAlgorithm

class DirichletDIAYNTorchOnlineRLAlgorithm(DIAYNTorchOnlineRLAlgorithm):
    def _train(self):
        self.training_mode(False)
        if self.min_num_steps_before_training > 0:
            self.expl_data_collector.collect_new_steps(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            init_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
            gt.stamp('initial exploration', unique=True)

        num_trains_per_expl_step = self.num_trains_per_train_loop // self.num_expl_steps_per_train_loop
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')

            # set policy skill for one epoch
            self.policy.alpha_update(epoch, int(0.9 * self.num_epochs))
            self.policy.skill_reset()

            for _ in range(self.num_train_loops_per_epoch):
                for _ in range(self.num_expl_steps_per_train_loop):
                    self.expl_data_collector.collect_new_steps(
                        self.max_path_length,
                        1,  # num steps
                        discard_incomplete_paths=False,
                    )
                    gt.stamp('exploration sampling', unique=False)

                    self.training_mode(True)
                    for _ in range(num_trains_per_expl_step):
                        train_data = self.replay_buffer.random_batch(
                            self.batch_size)
                        self.trainer.train(train_data)
                    gt.stamp('training', unique=False)
                    self.training_mode(False)

            new_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp('data storing', unique=False)

            self._end_epoch(epoch)
