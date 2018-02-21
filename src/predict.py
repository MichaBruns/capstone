import numpy as np
import glob
from sklearn.utils import shuffle
from config import *
# import utils.batch_loading as ub
import argparse
import os
import data
import VoxelNetTrainer
import net
import cv2
import glob

def str2bool(v):
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    directory = '/home/micha/Udacity/voxelnet/'
    parser = argparse.ArgumentParser(description='predict')

    parser.add_argument('-n', '--tag', type=str, nargs='?', default='unknown_tag',
                        help='set log tag')

    args = parser.parse_args()

    print('\n\n{}\n\n'.format(args))
    tag = args.tag
    if tag == 'unknown_tag':
        tag = input('Enter log tag : ')
        print('\nSet log tag :"%s" ok !!\n' % tag)

    dataset_dir = cfg.PREPROCESSED_DATA_SETS_DIR

    preprocess = data.Preprocess()

    predictor = None



    for file in os.listdir(directory):
        if file.endswith(".bin"):
            lidar = np.fromfile(os.path.join(directory, file), np.float32)
            lidar = lidar.reshape((-1, 4))

            x = lidar[:, 0].copy()
            y = lidar[:, 1].copy()


            lidar[:, 0] = y
            lidar[:, 1] = x
            top = preprocess.lidar_to_top(lidar)

            if not predictor:
                predictor = VoxelNetTrainer.VoxelNetPredictor(tag=tag,
                                                          top_shape=top.shape)

            top_exp = np.expand_dims(top, axis=0)
            proposals, scores, probs = predictor.predict(top_view=top_exp, rgb=None)
            nms_proposals, nms_scores = predictor.post_process(proposals, probs, minProb=0.1)

            top_img = data.draw_top_image(top).copy()
            topbox = net.processing.boxes3d.draw_box3d_on_top(top_img, nms_proposals[0][:10, :, :], scores=nms_scores[0])
            cv2.imwrite('output_'+ os.path.basename(file)+".png", topbox)
