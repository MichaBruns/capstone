import utils.boxes
import net.processing.boxes3d
import numpy as np
import glob
from sklearn.utils import shuffle
from config import *
# import utils.batch_loading as ub
import argparse
import os
from utils.training_validation_data_splitter import TrainingValDataSplitter
from utils.batch_loading import BatchLoading2 as BatchLoading
import net.rpn_target_op
from shapely.geometry import Polygon
import net.processing.boxes




train_n_val_dataset = [
            # '2011_09_26/2011_09_26_drive_0001_sync', # for tracking
            '2011_09_26/2011_09_26_drive_0002_sync',
            '2011_09_26/2011_09_26_drive_0005_sync',
            # '2011_09_26/2011_09_26_drive_0009_sync',
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
            # '2011_09_26/2011_09_26_drive_0095_sync',
            # '2011_09_26/2011_09_26_drive_0096_sync',
            # '2011_09_26/2011_09_26_drive_0104_sync',
            # '2011_09_26/2011_09_26_drive_0106_sync',
            # '2011_09_26/2011_09_26_drive_0113_sync',
            # '2011_09_26/2011_09_26_drive_0117_sync',
            '2011_09_26/2011_09_26_drive_0119_sync',
        ]

# shuffle bag list or same kind of bags will only be in training or validation set.
train_n_val_dataset = shuffle(train_n_val_dataset, random_state=666)
data_splitter = TrainingValDataSplitter(train_n_val_dataset)

with BatchLoading(tags=data_splitter.training_tags, require_shuffle=True, random_num=np.random.randint(100),
                  is_flip=False) as training:
    with BatchLoading(tags=data_splitter.val_tags, queue_size=1, require_shuffle=True,random_num=666) as validation:
        anchors, anchors3d = utils.boxes.create_anchors()
        anchors = np.asarray(anchors)
        batch_rgb_images, batch_top_view, batch_front_view, \
        batch_gt_labels, batch_gt_boxes3d, frame_id = \
            training.load()


        gt_top = net.processing.boxes3d.box3d_to_top_box(batch_gt_boxes3d[0])
        anchors_top = net.processing.boxes3d.box3d_to_top_box(anchors3d)
        validIds =  np.arange(0, len(anchors))
        pos_neg_inds, pos_inds, labels, targets, argmax_overlaps, idx_target = net.rpn_target_op.rpn_target(anchors_top, validIds, batch_gt_labels[0], gt_top, np.asarray(anchors), batch_gt_boxes3d[0])
        reprojectedBoxes = net.processing.boxes.box_transform_voxelnet_inv(targets, anchors[list(pos_inds)])
        error = (batch_gt_boxes3d[0][argmax_overlaps])[idx_target]- reprojectedBoxes

        pos_gt_boxes = (batch_gt_boxes3d[0][argmax_overlaps])[idx_target]

        # test for reconstruction error
        gtPolys = [Polygon(pos_gt_boxes[i, 0:4, 0:2]) for i in range(len(pos_gt_boxes))]
        predPolys = [Polygon(reprojectedBoxes[i, 0:4, 0:2]) for i in range(len(reprojectedBoxes))]

        errors = [predPolys[i].symmetric_difference(gtPolys[i]).area for i in range(len(predPolys))]

        pos_anchors = np.asarray(anchors)[idx_target]
        targets = net.processing.boxes.box_transform_voxelnet(pos_anchors, pos_gt_boxes)
        reprojectedBoxes = net.processing.boxes.box_transform_voxelnet_inv(targets, anchors[list(pos_inds)])
        here = True