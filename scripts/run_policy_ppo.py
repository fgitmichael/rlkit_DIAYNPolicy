from rlkit.samplers.util import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import joblib
import uuid
from rlkit.core import logger

filename = str(uuid.uuid4())


def simulate_policy(args):
    data = joblib.load(args.file)
    policy = data['evaluation/policy']
    env = data['evaluation/env']
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()

    import cv2
    video = cv2.VideoWriter('ppo_test.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640, 480))
    index = 0

    while True:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=False,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()

        for i, img in enumerate(path['images']):
            print(i)
            video.write(img[:,:,::-1].astype(np.uint8))
            cv2.imwrite("frames/ppo_test/%06d.png" % index, img[:,:,::-1])
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
