import numpy as np
import os
import glob
import argparse
from sklearn.utils import shuffle
from config import *
import net.processing.boxes3d
from utils.training_validation_data_splitter import TrainingValDataSplitter
from utils.batch_loading import BatchLoading2 as BatchLoading
import net.processing.boxes3d
import VoxelNetTrainer
import random


def str2bool(v):
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def writeGroundtruth(filename, obstacles, boxes3d, boxes2d):
    with open(filename, 'w') as f:
        for i, obs in enumerate(obstacles):
            x, y, z = np.sum(boxes3d[0, i,:,:], axis=0)/8

            f.write(obs.type + ' ' + str(obs.truncation) + ' ' + str(int(obs.occlusion)) + ' 0 '
                    + str(np.min(boxes2d[i, :, 0])) + " " + str(np.min(boxes2d[i, :, 1])) + " "
                    + str(np.max(boxes2d[i, :, 0])) + " " + str(
                np.max(boxes2d[i, :, 1])) + " "
                    + str(obs.size[0]) + " " + str(obs.size[1]) + " " + str(obs.size[2]) + " "
                    + str(x) + " " + str(y) + " " + str(z) + " "
                    + str(obs.rotation[2])
                    + "\n");

def writeDetections(filename, obstacles, scores, boxes2d):
    with open(filename, 'w') as f:
        for i, obs in enumerate(obstacles):
            Points0 = obs[0, 0:2]
            Points1 = obs[1, 0:2]
            Points2 = obs[2, 0:2]

            dis1 = np.sum((Points0 - Points1) ** 2) ** 0.5
            dis2 = np.sum((Points1 - Points2) ** 2) ** 0.5
            L = np.maximum(dis1, dis2)
            W = np.minimum(dis1, dis2)
            H = np.fabs(obs[0,2] - obs[4,2])
            x, y, z = np.sum(obs, axis=0)/8

            # assume that length > width
            t1 = np.linalg.norm(obs[1, :2] - obs[0, :2], axis=0)
            # gt_dist2 = np.linalg.norm(gt_boxes[:, 2, :2] - gt_boxes[:, 0, :2], axis=1)
            t2 = np.linalg.norm(obs[ 3, :2] - obs[0, :2], axis=0)

            diffX = obs[3, 0] - obs[0, 0]
            diffX2 = obs[1, 0] - obs[0, 0]
            diffX = np.where(t1 > t2, diffX2, diffX)
            inversed = np.where(diffX < 0)
            if diffX < 0:
                diffX *= -1
            diffY = obs[3, 1] - obs[0, 1]
            diffY2 = obs[1, 1] - obs[0, 1]
            diffY = np.where(t1 > t2, diffY2, diffY)
            if diffX < 0:
                diffY *= -1

            theta = np.arctan2(diffY, diffX)

            f.write('Car' + ' ' + "0" + ' ' + "0" + ' 0 '
                    + str(np.min(boxes2d[i, :, 0])) + " " + str(np.min(boxes2d[i, :, 1])) + " "
                    + str(np.max(boxes2d[i, :, 0])) + " " + str(
                np.max(boxes2d[i, :, 1])) + " "
                    + str(H) + " " + str(W) + " " + str(L) + " "
                    + str(x) + " " + str(y) + " " + str(z) + " "
                    + str(theta) + " " + str(float(scores[i]))
                    + "\n");

def writeTestDetections(filename, obstacles, boxes2d):
    with open(filename, 'w') as f:
        for i, obs in enumerate(obstacles):
            f.write(obs.type + ' ' + str(obs.truncation) + ' ' + str(obs.occlusion) + ' 0 '
                    + str(np.min(boxes2d[i, :, 0])) + " " + str(np.min(boxes2d[i, :, 1])) + " "
                    + str(np.max(boxes2d[i, :, 0])) + " " + str(
                np.max(boxes2d[i, :, 1])) + " "
                    + str(obs.size[0]) + " " + str(obs.size[1]) + " " + str(obs.size[2]) + " "
                    + str(obs.translation[0]) + " " + str(obs.translation[1]) + " " + str(obs.translation[2]) + " "
                    + str(obs.rotation[2]) + " " + str(random.random())
                    + "\n");

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate')

    parser.add_argument('-n', '--tag', type=str, nargs='?', default='unknown_tag',
                        help='set tag that will be evaluated')

    args = parser.parse_args()

    print('\n\n{}\n\n'.format(args))
    tag = args.tag

    print('\n\n{}\n\n'.format(args))

    train_n_val_dataset = [
        '2011_09_26/2011_09_26_drive_0001_sync',  # for tracking
        '2011_09_26/2011_09_26_drive_0002_sync',
        '2011_09_26/2011_09_26_drive_0005_sync',
        '2011_09_26/2011_09_26_drive_0009_sync',
        '2011_09_26/2011_09_26_drive_0011_sync',
        '2011_09_26/2011_09_26_drive_0013_sync',
        '2011_09_26/2011_09_26_drive_0014_sync',
        '2011_09_26/2011_09_26_drive_0015_sync',
        '2011_09_26/2011_09_26_drive_0017_sync',
        '2011_09_26/2011_09_26_drive_0018_sync',
        '2011_09_26/2011_09_26_drive_0019_sync',
        '2011_09_26/2011_09_26_drive_0020_sync',
        '2011_09_26/2011_09_26_drive_0022_sync',
        '2011_09_26/2011_09_26_drive_0023_sync',
        '2011_09_26/2011_09_26_drive_0027_sync',
        '2011_09_26/2011_09_26_drive_0028_sync',
        '2011_09_26/2011_09_26_drive_0029_sync',
        '2011_09_26/2011_09_26_drive_0032_sync',
        '2011_09_26/2011_09_26_drive_0035_sync',
        '2011_09_26/2011_09_26_drive_0036_sync',
        '2011_09_26/2011_09_26_drive_0039_sync',
        '2011_09_26/2011_09_26_drive_0046_sync',
        '2011_09_26/2011_09_26_drive_0048_sync',
        '2011_09_26/2011_09_26_drive_0051_sync',
        '2011_09_26/2011_09_26_drive_0052_sync',
        '2011_09_26/2011_09_26_drive_0056_sync',
        '2011_09_26/2011_09_26_drive_0057_sync',
        '2011_09_26/2011_09_26_drive_0059_sync',
        '2011_09_26/2011_09_26_drive_0060_sync',
        '2011_09_26/2011_09_26_drive_0061_sync',
        '2011_09_26/2011_09_26_drive_0064_sync',
        '2011_09_26/2011_09_26_drive_0070_sync',
        '2011_09_26/2011_09_26_drive_0079_sync',
        '2011_09_26/2011_09_26_drive_0084_sync',
        '2011_09_26/2011_09_26_drive_0086_sync',
        '2011_09_26/2011_09_26_drive_0087_sync',
        '2011_09_26/2011_09_26_drive_0091_sync',
        # '2011_09_26/2011_09_26_drive_0093_sync',  #data size not same
    ]

    # shuffle bag list or same kind of bags will only be in training or validation set.
    train_n_val_dataset = shuffle(train_n_val_dataset, random_state=666)
    data_splitter = TrainingValDataSplitter(train_n_val_dataset)

    outputPath = "eval"
    #tag = "bla3"
    gtDir = os.path.join(outputPath, "data", "object", "label_2")
    detDir = os.path.join(outputPath, "results", tag, "data")
    os.makedirs(gtDir, exist_ok=True)
    os.makedirs(detDir, exist_ok=True)

    with open(outputPath + "/lists/"+tag+".txt", 'w') as filelist:
        with BatchLoading(tags=data_splitter.training_tags, require_shuffle=True, random_num=np.random.randint(100),
                          is_flip=False) as training:
            with BatchLoading(tags=data_splitter.val_tags, queue_size=16, require_shuffle=False,
                              random_num=666) as validation:
                voxelnet = VoxelNetTrainer.VoxelNetTrainer(training_set=training, validation_set=validation, tag=tag,
                                                    continue_train=True)

                # iterate through validation set
                for i in range(len(data_splitter.val_tags)):
                    print("Iteration {} from {}".format(i, len(data_splitter.val_tags)))
                    rgb_image, top_view, front_view, \
                    gt_labels, gt_boxes3d, frame_id, obstacles = \
                        validation.load()

                    print(frame_id)

                    boxes2d = net.processing.boxes3d.box3d_to_rgb_box(gt_boxes3d[0,:])
                    # write gt to file for evaluation
                    filename = "%06d" % i + ".txt"
                    filelist.write(str(filename) + "\n")
                    filelist.flush()
                    writeGroundtruth(gtDir + "/" + filename, obstacles, gt_boxes3d, boxes2d)

                    #writeTestDetections(detDir + "/" + filename, obstacles, boxes2d)

                    proposals, scores, probs, conv = voxelnet.predict(top_view)

                    nms_proposals, nms_scores = voxelnet.post_process(proposals, probs, minProb=-1)

                    #nms_proposals = np.expand_dims(nms_proposals, axis=0)

                    boxes2d = net.processing.boxes3d.box3d_to_rgb_box(nms_proposals[0])
                    writeDetections(detDir + "/" + filename, nms_proposals[0], scores=nms_scores[0], boxes2d= boxes2d)