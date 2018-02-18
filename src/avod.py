import net.utility.draw  as nud
import tensorflow as tf
import net.layers
import utils.boxes
import numpy as np
import os
import subprocess
import net.rpn_target_op
import net.rpn_nms_op
import config as cfg

conv_net_name = 'vgg16'
rpn_name = 'fusion'



class AVOD():
    def __init__(self):
        self.name = 'AVOD'
        self.conv_net_name = conv_net_name
        self.rpn_name = rpn_name


    def create_anchors(self, step):
        self.anchors, self.anchors3d = utils.boxes.create_anchors(step)
        XSIZE = cfg.TOP_X_SIZE
        YSIZE = cfg.TOP_Y_SIZE
        self.anchorsOut = self.anchors.reshape(int(XSIZE//step), int(YSIZE//step), 2,7)
        self.anchors_top = net.processing.boxes3d.box3d_to_top_box(self.anchors3d)

    def build_net(self, top_shape, rgb_shape):
        self.create_anchors(2)

        with tf.variable_scope(self.name):
            topview, rgb = self.create_input(top_shape, rgb_shape)
            with tf.variable_scope(conv_net_name):
                top_features = self.create_vgg16(topview, name_scope='lidar')
                rgb_features = self.create_vgg16(rgb, name_scope='rgb')
            probs, scores, proposals = self.create_rpn(top_features, rgb_features, boxes=self.anchors_top)

        with tf.variable_scope('loss'):
            self.cls_loss, self.reg_loss, self.target_loss = \
                self.create_loss(scores, proposals, self.top_inds, self.top_pos_inds, self.top_labels, self.top_targets)

        return {
            'top_view' : self.top_view,
            'top_inds':self.top_inds,
            'top_pos_inds': self.top_pos_inds,
            'rgb': self.rgb,
            'labels':self.top_labels,
            'targets':self.top_targets,
            'cls_loss': self.cls_loss,
            'reg_loss': self.reg_loss,
            'target_loss': self.target_loss,
            'proposals': proposals,
            'scores': scores,
            'probs':probs
        }

    def get_targets(self, batch_gt_labels, batch_gt_boxes3d):
        """

        :param batch_gt_labels: Labels (0 or 1) for bounding boxes
        :param batch_gt_boxes3d:  ground truth 3d bounding boxes
        :return:
             pos_neg_inds : positive and negative samples
             pos_inds : positive samples
             labels: pos_neg_inds's labels
             targets:  positive samples's bias to ground truth (top view bounding box regression targets)
        """
        gt_top = net.processing.boxes3d.box3d_to_top_box(batch_gt_boxes3d[0])

        # for now, every box is valid
        validIds = np.arange(0, len(self.anchors), dtype=np.uint32)
        pos_neg_inds, pos_inds, labels, targets,_, _ = net.rpn_target_op.rpn_target(self.anchors_top, validIds,
                                                                               batch_gt_labels[0], gt_top,
                                                                               self.anchors, batch_gt_boxes3d[0])
        return pos_neg_inds, pos_inds, labels, targets

    def create_input(self, top_shape, rgb_shape):
        self.top_view = tf.placeholder(shape=[None, *top_shape], dtype=tf.float32, name='top')
        self.rgb = tf.placeholder(shape=[None, *rgb_shape], dtype=tf.float32, name='rgb')
        with tf.variable_scope('lossVars'):
            self.top_inds = tf.placeholder(shape=[None], dtype=tf.int32, name='top_ind')
            self.top_pos_inds = tf.placeholder(shape=[None], dtype=tf.int32, name='top_pos_ind')
            self.top_labels = tf.placeholder(shape=[None], dtype=tf.int32, name='top_label')
            self.top_targets = tf.placeholder(shape=[None, 7], dtype=tf.float32, name='top_target')

        return self.top_view, self.rgb

    def create_vgg16(self, input, name_scope):
        """
        Create the a modified vgg16 encoder
        :param input: Input tensor
        :return:
        """
        with tf.variable_scope(name_scope):
            maxpoolLayer = tf.layers.MaxPooling2D((2, 2), 2, 'same', name='MaxPool')

            # first block
            conv = net.layers.conv2d(input, filters=32, kernel_size=3, strides=(1, 1), padding='same',
                                     name='Conv2D_1_1')
            conv = net.layers.conv2d(conv, filters=32, kernel_size=3, strides=(1, 1), padding='same',
                                     name='Conv2D_1_3')
            maxpool = maxpoolLayer(conv)

            # second block
            conv = net.layers.conv2d(maxpool, filters=64, kernel_size=3, strides=(1, 1), padding='same',
                                     name='Conv2D_2_1')
            conv = net.layers.conv2d(conv, filters=64, kernel_size=3, strides=(1, 1), padding='same',
                                     name='Conv2D_2_3')
            maxpool = maxpoolLayer(conv)

            # third block
            conv = net.layers.conv2d(maxpool, filters=128, kernel_size=3, strides=(1, 1), padding='same',
                                     name='Conv2D_3_1')
            conv = net.layers.conv2d(conv, filters=128, kernel_size=3, strides=(1, 1), padding='same',
                                     name='Conv2D_3_2')
            conv = net.layers.conv2d(conv, filters=128, kernel_size=3, strides=(1, 1), padding='same',
                                     name='Conv2D_3_3')
            maxpool = maxpoolLayer(conv)

            # fourth block
            conv = net.layers.conv2d(maxpool, filters=256, kernel_size=3, strides=(1, 1), padding='same',
                                     name='Conv2D_4_1')
            conv = net.layers.conv2d(conv, filters=256, kernel_size=3, strides=(1, 1), padding='same',
                                     name='Conv2D_4_2')
            conv = net.layers.conv2d(conv, filters=256, kernel_size=3, strides=(1, 1), padding='same',
                                     name='Conv2D_4_3')
            maxpool = maxpoolLayer(conv)

            # fifth block
            conv = net.layers.conv2d(maxpool, filters=256, kernel_size=3, strides=(1, 1), padding='same',
                                     name='Conv2D_5_1')
            conv = net.layers.conv2d(conv, filters=256, kernel_size=3, strides=(1, 1), padding='same',
                                     name='Conv2D_5_2')
            conv = net.layers.conv2d(conv, filters=256, kernel_size=3, strides=(1, 1), padding='same',
                                     name='Conv2D_5_3')

            conv_shape = conv.get_shape().as_list()

            upsample = tf.image.resize_bilinear(conv, size=[conv_shape[1]*2, conv_shape[2]*2])


        return upsample

    def create_rpn(self, inputImage, inputLidar, boxes):
        feature_maps = []
        with tf.variable_scope(rpn_name):
            with tf.variable_scope('Bottleneck'):
                bottlneck_image = tf.layers.conv2d(inputImage, filters=1, kernel_size=1, strides=(1, 1),
                                                   padding='valid')
                bottlneck_lidar = tf.layers.conv2d(inputLidar, filters=1, kernel_size=1, strides=(1, 1),
                                                   padding='valid')
                cr_image = tf.image.crop_and_resize(bottlneck_image, boxes=boxes,
                                                    box_ind=tf.zeros(boxes.shape[0], dtype=tf.int32), crop_size=[3, 3])
                cr_lidar = tf.image.crop_and_resize(bottlneck_lidar, boxes=boxes,
                                                    box_ind=tf.zeros(boxes.shape[0], dtype=tf.int32), crop_size=[3, 3])


            with tf.variable_scope('Fusion'):
                concat = tf.concat([cr_image, cr_lidar] , axis=-1, name='Concat')
                fused = tf.reduce_mean(concat, axis=-1, keep_dims=True, name='Mean')


            with tf.variable_scope('Regression'):
                with tf.variable_scope('Scores') as scope:
                    # create scores for each anchor and each class
                    fused = tf.reshape(fused, [-1, 3*3])
                    scores = tf.layers.dense(fused, units=256)
                   # scores = tf.layers.conv2d(dense, filters=cfg.cfg.NUM_CLASSES * 2, kernel_size=1, strides=(1, 1), padding='valid')

                scores = tf.layers.dense(scores, units=cfg.cfg.NUM_CLASSES)
                probs = tf.nn.softmax(scores, name='Probabilities')

                # 7xnum_anchors
                reg = tf.layers.dense(fused, units=7,  name='Map')
                reg = tf.reshape(reg, [-1, 7])

            return probs, scores ,reg

    def create_loss(self, scores, deltas, inds, pos_inds, gt_labels, gt_targets):
        """

        :param scores: Softmax output of predictor
        :param deltas: Regression output. Delta to corresponding anchors
        :param inds: Indices of positive and negative anchors
        :param pos_inds: Indices of positive anchors
        :param gt_labels: Labels of anchors
        :param gt_targets: Groundtruth delta for anchors
        :return:
                Classification loss, regression loss and target loss
        """

        def modified_smooth_l1( box_preds, box_targets, sigma=3.0):
            '''
                ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
                SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                              |x| - 0.5 / sigma^2,    otherwise
            '''
            sigma2 = sigma * sigma
            diffs  =  tf.subtract(box_preds, box_targets)
            smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

            smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
            smooth_l1_option2 = tf.abs(diffs) - 0.  / sigma2
            smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + tf.multiply(smooth_l1_option2, 1-smooth_l1_signs)
            smooth_l1 = smooth_l1_add   #tf.multiply(box_weights, smooth_l1_add)  #

            return smooth_l1

        rpn_scores   = tf.gather(scores,inds)  # remove ignore label
        rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_scores,
                                                                                      labels=gt_labels))

        deltas1       = deltas
        rpn_deltas    = tf.gather(deltas1, pos_inds)  # remove ignore label

        with tf.variable_scope('modified_smooth_l1'):
            rpn_smooth_l1 = modified_smooth_l1(rpn_deltas, gt_targets, sigma=3.0)

        rpn_reg_loss  = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, axis=1))

        target_loss = rpn_cls_loss + rpn_reg_loss
        return rpn_cls_loss, rpn_reg_loss, target_loss
