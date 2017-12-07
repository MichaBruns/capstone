import net.utility.draw  as nud
import tensorflow as tf
import net.layers
import utils.boxes
import numpy as np

CONV_NET_NAME = 'MidConv'

class VoxelNet():
    def __init__(self, train_set, validation_set, pre_trained_weights, train_targets, log_tag=None,
                 continue_train=False, fast_test_mode=False):

        self.build_net((400, 352, 10), (400, 352, 10))
        self.batch_rgb_images, self.batch_top_view, self.batch_front_view, \
        self.batch_gt_labels, self.batch_gt_boxes3d, self.frame_id = \
            train_set.load()
        top_boxes = net.processing.boxes3d.box3d_to_top_box(self.batch_gt_boxes3d[0])
        h = True


    def create_anchors(self, step):
        self.anchors, self.anchors3d = utils.boxes.create_anchors(step)
        self.anchors = np.asarray(self.anchors)
        self.anchors_top = net.processing.boxes3d.box3d_to_top_box(self.anchors3d)

    def build_net(self, top_shape, rgb_shape):
        topview = self.create_input(top_shape, rgb_shape)
        conv = self.create_convnet_hc(topview)
        probs, reg = self.create_rpn(conv)
        factor = top_shape[0] / probs.get_shape().as_list()[1]
        self.create_anchors(factor)

    def get_targets(self, batch_gt_labels, batch_gt_boxes3d):
        gt_top = net.processing.boxes3d.box3d_to_top_box(batch_gt_boxes3d[0])
        validIds = np.arange(0, len(self.anchors))
        pos_neg_inds, pos_inds, labels, targets = net.rpn_target_op.rpn_target(self.anchors_top, validIds,
                                                                               batch_gt_labels[0], gt_top,
                                                                               self.anchors, batch_gt_boxes3d[0])
        return pos_neg_inds, pos_inds, labels, targets

    def create_input(self, top_shape, rgb_shape):
        self.top_view = tf.placeholder(shape=[None, *top_shape], dtype=tf.float32, name='top')
        #self.rgb_images = tf.placeholder(shape=[None, *rgb_shape], dtype=tf.float32, name='rgb')
        return self.top_view

    """
    Creates the convolutional middle layers
    """
    def create_convnet(self, input):
        net.layers.conv3d(input)


    """
    Creates the convolutional middle layers
    """
    def create_convnet_hc(self, input):
        conv = net.layers.conv2d(input, filters=32, kernel_size=3, strides=(1,1), padding='same', name='Conv2D_1')
        conv = net.layers.conv2d(conv, filters=64, kernel_size=3, strides=(2, 2), padding='same', name='Conv2D_2')
        conv = net.layers.conv2d(conv, filters=128, kernel_size=3, strides=(1, 1), padding='same', name='Conv2D_3')

        return conv

    def create_rpn(self, input):
        feature_maps = []
        input_shape = input.get_shape().as_list()
        with tf.variable_scope('RPN'):
            with tf.variable_scope('Block1'):
                conv = net.layers.conv2d(input, filters=128, kernel_size=3, strides=(2, 2), padding='same', name='Conv2D_1')
                conv = net.layers.conv2d(conv, filters=128, kernel_size=3, strides=(1, 1), padding='same', name='Conv2D_2')
                conv = net.layers.conv2d(conv, filters=128, kernel_size=3, strides=(1, 1), padding='same', name='Conv2D_3')
                feature1 = net.layers.conv2d(conv, filters=128, kernel_size=3, strides=(1, 1), padding='same', name='Conv2D_4')

            with tf.variable_scope('Block2'):
                conv = net.layers.conv2d(feature1, filters=128, kernel_size=3, strides=(2, 2), padding='same', name='Conv2D_1')
                conv = net.layers.conv2d(conv, filters=128, kernel_size=3, strides=(1, 1), padding='same', name='Conv2D_2')
                conv = net.layers.conv2d(conv, filters=128, kernel_size=3, strides=(1, 1), padding='same', name='Conv2D_3')
                conv = net.layers.conv2d(conv, filters=128, kernel_size=3, strides=(1, 1), padding='same', name='Conv2D_4')
                conv = net.layers.conv2d(conv, filters=128, kernel_size=3, strides=(1, 1), padding='same', name='Conv2D_5')
                feature2 = net.layers.conv2d(conv, filters=128, kernel_size=3, strides=(1, 1), padding='same', name='Conv2D_6')

            with tf.variable_scope('Block3') as scope2:
                conv = net.layers.conv2d(feature2, filters=256, kernel_size=3, strides=(2, 2), padding='same', name='Conv2D_1')
                conv = net.layers.conv2d(conv, filters=256, kernel_size=3, strides=(1, 1), padding='same', name='Conv2D_2')
                conv = net.layers.conv2d(conv, filters=256, kernel_size=3, strides=(1, 1), padding='same', name='Conv2D_3')
                conv = net.layers.conv2d(conv, filters=256, kernel_size=3, strides=(1, 1), padding='same', name='Conv2D_4')
                conv = net.layers.conv2d(conv, filters=256, kernel_size=3, strides=(1, 1), padding='same', name='Conv2D_5')
                feature3 = net.layers.conv2d(conv, filters=256, kernel_size=3, strides=(1, 1), padding='same', name='Conv2D_6')

            with tf.variable_scope('Fusion'):
                #<todo> make shape dynamic
                upsample1 = net.layers.deconv2d(feature1,
                                                [int(input_shape[1]/2), int(input_shape[2]/2), 256, 128],
                                                (-1, int(input_shape[1]/2), int(input_shape[2]/2), 256),
                                                [1, 1, 1, 1], padding='SAME', name='Upsample1')
                upsample2 = net.layers.deconv2d(feature2,
                                                [int(input_shape[1]/2), int(input_shape[2]/2), 256, 128],
                                                (-1, int(input_shape[1]/2), int(input_shape[2]/2), 256),
                                                [1, 1, 1, 1], padding='SAME', name='Upsample2')
                upsample3 = net.layers.deconv2d(feature3,
                                                [int(input_shape[1]/2), int(input_shape[2]/2), 256, 256],
                                                (-1, int(input_shape[1]/2), int(input_shape[2]/2), 256),
                                                [1, 1, 1, 1], padding='SAME', name='Upsample3')

                feature_maps.append(upsample1)
                feature_maps.append(upsample2)
                feature_maps.append(upsample3)
                concat = net.layers.concat(feature_maps)

            with tf.variable_scope('Regression'):
                probs = net.layers.conv2d(concat, filters=2, kernel_size=1, strides=(1, 1), padding='valid',
                                         name='Probabilities')

                # 7xnum_anchors
                reg = net.layers.conv2d(concat, filters=14, kernel_size=1, strides=(1, 1), padding='valid',
                                         name='Map')

            return probs, reg

    def create_loss(self, scores, deltas, inds, pos_inds, rpn_labels, rpn_targets):

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

        scores1      = tf.reshape(scores,[-1,2])
        rpn_scores   = tf.gather(scores1,inds)  # remove ignore label
        rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_scores, labels=rpn_labels))

        deltas1       = tf.reshape(deltas,[-1,4])
        rpn_deltas    = tf.gather(deltas1, pos_inds)  # remove ignore label

        with tf.variable_scope('modified_smooth_l1'):
            rpn_smooth_l1 = modified_smooth_l1(rpn_deltas, rpn_targets, sigma=3.0)

        rpn_reg_loss  = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, axis=1))
        return rpn_cls_loss, rpn_reg_loss