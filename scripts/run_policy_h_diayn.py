from rlkit.samplers.util import hierarchicalRollout as rollout
from rlkit.torch.pytorch_util import set_gpu_mode
from rlkit.envs.wrappers import NormalizedBoxEnv
import argparse
import joblib
import uuid
from rlkit.core import logger
import numpy as np

filename = str(uuid.uuid4())


def simulate_policy(args):
    manager_data = joblib.load(args.manager_file)
    worker_data = joblib.load(args.worker_file)
    policy = manager_data['evaluation/policy']
    worker = worker_data['evaluation/policy']
    env = NormalizedBoxEnv(gym.make("BipedalWalker-v2"))
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()

    import cv2
    video = cv2.VideoWriter('h_diayn_test.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640, 480))
    index = 0

    path = rollout(
        env,
        policy,
        worker,
        continuous=True,
        max_path_length=args.H,
        render=True,
    )
    if hasattr(env, "log_diagnostics"):
        env.log_diagnostics([path])
    logger.dump_tabular()

    for i, img in enumerate(path['images']):
        print(i)
        video.write(img[:,:,::-1].astype(np.uint8))
        cv2.imwrite("frames/h_diayn_test/%06d.png" % index, img[:,:,::-1])
        index += 1

    video.release()
    print("wrote video")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('manager_file', type=str,
                        help='path to the manager snapshot file')
    parser.add_argument('worker_file', type=str,
                        help='path to the worker snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
