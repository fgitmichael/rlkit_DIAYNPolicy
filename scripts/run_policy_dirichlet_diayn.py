from rlkit.samplers.util import DIAYNRollout as rollout
from rlkit.torch.pytorch_util import set_gpu_mode
from rlkit.envs.wrappers import NormalizedBoxEnv
import gym
import torch
import argparse
#import joblib
import uuid
from rlkit.core import logger
import numpy as np
from torch.distributions.dirichlet import Dirichlet

skill_dim = 10
dirichlet = Dirichlet(torch.ones(skill_dim))

filename = str(uuid.uuid4())


def simulate_policy(args):
 #   data = joblib.load(args.file)
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    env = NormalizedBoxEnv(gym.make("BipedalWalker-v2"))
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()

    import cv2
    video = cv2.VideoWriter('dirichlet_diayn_test.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640, 480))
    index = 0
    for z in range(policy.stochastic_policy.skill_dim*2):
        if z < policy.stochastic_policy.skill_dim:
            skill = np.zeros(policy.stochastic_policy.skill_dim)
            skill[z] = 1
        else:
            skill = dirichlet.sample().cpu().numpy()
        for _ in range(3):
            path = rollout(
                env,
                policy,
                skill,
                max_path_length=args.H,
                render=True,
            )
            if hasattr(env, "log_diagnostics"):
                env.log_diagnostics([path])
            logger.dump_tabular()

            for i, img in enumerate(path['images']):
                print(i)
                video.write(img[:,:,::-1].astype(np.uint8))
                cv2.imwrite("frames/dirichlet_diayn_test/%06d.png" % index, img[:,:,::-1])
                index += 1

    video.release()
    print("wrote video")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
