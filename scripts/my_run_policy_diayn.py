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

def simulate_policy(args):
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    env = NormalizedBoxEnv(gym.make("HalfCheetah-v2"))
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()

    import cv2
    video_name = 'HalfCheetah-v2_diayn_policies.avi'
    video = cv2.VideoWriter(video_name,
                            cv2.VideoWriter_fourcc('M','J','P','G'),
                            30, (500, 500))
    index = 0
    for skill in range(policy.stochastic_policy.skill_dim):
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
                print(img.shape)
                bgr_img = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_RGB2BGR)
                video.write(bgr_img)
#                cv2.imwrite("frames/diayn_bipedal_walker_hardcore.avi/%06d.png" % index, img[:,:,::-1])
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

