import voxelnet
import numpy as np
import tensorflow as tf
import subprocess
import os
import cv2
from config import *
import net.processing.boxes3d
import net.layers
from time import localtime, strftime
import data
import io
import pickle
import matplotlib.pyplot as plt
import net.utility.draw as nud
import net.rpn_nms_op

class Net(object):

    def __init__(self, prefix, scope_name, checkpoint_dir=None):
        self.name =scope_name
        self.prefix = prefix
        self.checkpoint_dir =checkpoint_dir
        self.subnet_checkpoint_dir = os.path.join(checkpoint_dir, scope_name)
        self.subnet_checkpoint_name = scope_name
        os.makedirs(self.subnet_checkpoint_dir, exist_ok=True)
        self.variables = self.get_variables([prefix+'/'+scope_name])
        self.saver=  tf.train.Saver(self.variables)


    def save_weights(self, sess=None, dir=None):
        path = os.path.join(dir, self.subnet_checkpoint_name)
        print('\nSave weigths : %s' % path)
        os.makedirs(dir,exist_ok=True)
        self.saver.save(sess, path)

    def clean_weights(self):
        command = 'rm -rf %s' % (os.path.join(self.subnet_checkpoint_dir))
        subprocess.call(command, shell=True)
        print('\nClean weights: %s' % command)
        os.makedirs(self.subnet_checkpoint_dir ,exist_ok=True)


    def load_weights(self, sess=None):
        path = os.path.join(self.subnet_checkpoint_dir, self.subnet_checkpoint_name)
        if tf.train.checkpoint_exists(path) ==False:
            print('\nCan not found :\n"%s",\nuse default weights instead it\n' % (path))
            path = path.replace(os.path.basename(self.checkpoint_dir),'default')
        assert tf.train.checkpoint_exists(path) == True
        self.saver.restore(sess, path)


    def get_variables(self, scope_names):
        variables=[]
        for scope in scope_names:
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            assert len(variables) != 0
            variables += variables
        return variables


class VoxelNetTrainer():
    def __init__(self, training_set, validation_set, tag, continue_train=False):
        self.train_set = training_set
        self.validation_set = validation_set
        self.ckpt_dir = os.path.join(cfg.CHECKPOINT_DIR, tag)
        self.tb_dir = tag if tag != None else strftime("%Y_%m_%d_%H_%M", localtime())
        self.tag = tag
        self.n_global_step = 0

        top_shape, _, _ = training_set.get_shape()
        self.voxelnet = voxelnet.VoxelNet()
        self.placeholder = self.voxelnet.build_net(top_shape, top_shape)

        tf.summary.scalar('cls_loss', self.placeholder['cls_loss'])
        tf.summary.scalar('reg_loss', self.placeholder['reg_loss'])
        tf.summary.scalar('target_loss', self.placeholder['target_loss'])

        train_targets = [voxelnet.conv_net_name, voxelnet.rpn_name]

        self.subnet_conv = Net(prefix='VoxelNet', scope_name=voxelnet.conv_net_name, checkpoint_dir=self.ckpt_dir)
        self.subnet_rpn = Net(prefix='VoxelNet', scope_name=voxelnet.rpn_name, checkpoint_dir=self.ckpt_dir)


        self.sess = tf.Session()
        with self.sess.as_default():
            with tf.variable_scope('minimize_loss'):
                solver = tf.train.GradientDescentOptimizer(learning_rate=0.001)

                train_var_list = []

                assert train_targets != []
                for target in train_targets:
                    # variables
                    if target == voxelnet.conv_net_name:
                        train_var_list += self.subnet_conv.variables

                    elif target == voxelnet.rpn_name:
                        train_var_list += self.subnet_rpn.variables

                    else:
                        ValueError('unknow train_target name')

                self.solver_step = solver.minimize(self.voxelnet.target_loss, var_list=train_var_list)
                self.sess.run(tf.global_variables_initializer(),
                          {net.layers.IS_TRAIN_PHASE: True})

        train_writer_dir = os.path.join(cfg.LOG_DIR, 'tensorboard', self.tb_dir + '_train')
        val_writer_dir = os.path.join(cfg.LOG_DIR, 'tensorboard', self.tb_dir + '_val')
        graph = None if continue_train else tf.get_default_graph()
        self.train_summary_writer = tf.summary.FileWriter(train_writer_dir, graph=graph)
        self.val_summary_writer = tf.summary.FileWriter(val_writer_dir, graph=graph)

        if continue_train:
            self.load_weights([voxelnet.conv_net_name, voxelnet.rpn_name])
            self.load_progress()
            print("Restoring weights from iteration ", self.n_global_step)

        summ = tf.summary.merge_all()
        self.summ = summ

    def save_progress(self):
        print('Save progress !')
        path = os.path.join(cfg.LOG_DIR, 'train_progress',self.tag,'progress.data')
        os.makedirs(os.path.dirname(path) ,exist_ok=True)
        pickle.dump(self.n_global_step, open(path, "wb"))


    def load_progress(self):
        path = os.path.join(cfg.LOG_DIR, 'train_progress', self.tag, 'progress.data')
        if os.path.isfile(path):
            print('\nLoad progress !')
            self.n_global_step = pickle.load(open(path, 'rb'))
        else:
            print('\nCan not found progress file')

    def validate(self, maxIter):
        for iter in range(maxIter):
            print("Validating {} from {}".format(iter, maxIter))
            # get the data
            self.batch_rgb_images, self.batch_top_view, self.batch_front_view, \
            self.batch_gt_labels, self.batch_gt_boxes3d, self.frame_id = \
                self.validation_set.load()

            self.save_projection('validation')

    def train(self, maxIter):
        for iter in range(maxIter):
            if iter % 1000 == 0:
                cls_loss, reg_loss, target_loss = self.iteration(self.n_global_step, validation=True, writeSummary=True)

                print("Validation Iteration: {}/{}\nGlobal Iteration: {}".format(iter, maxIter, self.n_global_step))
                print("Loss: ", target_loss)


            if self.n_global_step % 1000 == 0:
                trainSummary = True
            else:
                trainSummary = False

            cls_loss, reg_loss, target_loss = self.iteration(self.n_global_step, validation=False, writeSummary=trainSummary)


            if self.n_global_step % 100 == 0:
                print("Iteration: {}/{}\nGlobal Iteration: {}".format(iter, maxIter, self.n_global_step))
                print("Loss: ", target_loss)

            if self.n_global_step % 10000 == 0 and self.n_global_step != 0:
                self.save_weights()
                self.save_progress()

            self.n_global_step+=1

    def save_projection(self, dir=None):
        proposals, scores, probs, conv = self.predict(self.batch_top_view)
        nms_proposals, nms_scores = self.post_process(proposals, probs, minProb=0.3)
        prediction_on_rgb = nud.draw_box3d_on_camera(self.batch_rgb_images[0], nms_proposals,
                                                     text_lables=[])
        cv2.imwrite(dir + '/' + self.frame_id.replace('/','_') + '.png', prediction_on_rgb)


    def iteration_summary(self, prefix, summaryWriter):
        # get the indices of pos+neg, positive anchors as well as labels and regression targets
        top_inds, pos_inds, labels, targets = self.voxelnet.get_targets(self.batch_gt_labels, self.batch_gt_boxes3d)
        top = data.draw_top_image(self.batch_top_view[0]).copy()

        self.log_image(step=self.n_global_step, prefix=prefix, frame_tag=self.frame_id, summary_writer=summaryWriter)
        proposals, scores, probs, conv = self.predict(self.batch_top_view)
        nms_proposals, nms_scores = self.post_process(proposals, probs, minProb=-1)

        self.summary_image(scores[0, :, :, 1], prefix + '/proposalMap', summaryWriter,
                           step=self.n_global_step)
        self.summary_image(probs[0, :, :, 0], prefix + '/probabilities0', summaryWriter,
                           step=self.n_global_step)
        self.summary_image(probs[0, :, :, 1], prefix + '/probabilities1', summaryWriter,
                           step=self.n_global_step)

        # print proposals
        proposalsFlat = proposals.reshape(-1, 7)
        proposalIdx = (probs > 0.2).flatten()
        numProposals = np.sum(proposalIdx)
        Boxes3d = []
        if (numProposals > 0):
            Boxes3d = net.processing.boxes.box_transform_voxelnet_inv(proposalsFlat[proposalIdx],
                                                                      self.voxelnet.anchors[proposalIdx])
        topbox = net.processing.boxes3d.draw_box3d_on_top(top, Boxes3d)
        self.summary_image(topbox, prefix + '/proposalBoxes', summaryWriter, step=self.n_global_step)

        # top 10 proposals

        sortedIdx = np.argsort(-probs.flatten())
        top10Proposals = proposalsFlat[sortedIdx[:10]]

        ### top visualization ###
        Boxes3d = net.processing.boxes.box_transform_voxelnet_inv(top10Proposals,
                                                                      self.voxelnet.anchors[sortedIdx[:10]])
        topbox = net.processing.boxes3d.draw_box3d_on_top(top, Boxes3d)
        self.summary_image(topbox, prefix + '/top10proposalBoxes', summaryWriter, step=self.n_global_step)



        topbox = net.processing.boxes3d.draw_box3d_on_top(top, nms_proposals[:10,:,:], scores=nms_scores)
        self.summary_image(topbox, prefix + '/top10NMSBoxes', summaryWriter, step=self.n_global_step)


        ### to rgb image ###
        prediction_on_rgb = nud.draw_box3d_on_camera(self.batch_rgb_images[0], nms_proposals[:10,:,:],
                                                     text_lables=[])
        self.summary_image(prediction_on_rgb, prefix + '/prediction_on_rgb', summaryWriter, step=self.n_global_step)

        ### Groundtruth ###
        Boxes3d = net.processing.boxes.box_transform_voxelnet_inv(targets, self.voxelnet.anchors[pos_inds])
        topbox = net.processing.boxes3d.draw_box3d_on_top(top, Boxes3d)
        self.summary_image(topbox, prefix + '/Targets', summaryWriter, step=self.n_global_step)

    def iteration(self, step, validation, writeSummary):
        if validation:
            summaryWriter = self.val_summary_writer
            dataset = self.validation_set
            prefix='validation'

        else:
            summaryWriter = self.train_summary_writer
            dataset = self.train_set
            prefix = 'training'

        # get the data
        self.batch_rgb_images, self.batch_top_view, self.batch_front_view, \
        self.batch_gt_labels, self.batch_gt_boxes3d, self.frame_id = \
            dataset.load()
        # get the indices of pos+neg, positive anchors as well as labels and regression targets

        top_inds, pos_inds, labels, targets = self.voxelnet.get_targets(self.batch_gt_labels, self.batch_gt_boxes3d)
        top = data.draw_top_image(self.batch_top_view[0]).copy()


        fd1 = {
            self.placeholder['top_view']: self.batch_top_view,
            self.placeholder['top_inds']: top_inds,
            self.placeholder['top_pos_inds']: pos_inds,
            self.placeholder['labels']: labels,  # ,[top_inds],
            self.placeholder['targets']: targets,
            net.layers.IS_TRAIN_PHASE: not validation
        }

        if validation:
            # without optimizer
            cls_loss, reg_loss, target_loss, summary = self.sess.run([self.placeholder['cls_loss'],
                                                                     self.placeholder['reg_loss'],
                                                                     self.placeholder['target_loss'], self.summ], fd1)
        else:
            _, cls_loss, reg_loss, target_loss, summary = self.sess.run([self.solver_step, self.placeholder['cls_loss'],
                                                                         self.placeholder['reg_loss'],
                                                                         self.placeholder['target_loss'], self.summ],
                                                                        fd1)

        summaryWriter.add_summary(summary, step)

        if writeSummary:
            self.iteration_summary(prefix=prefix, summaryWriter=summaryWriter)

        return cls_loss, reg_loss, target_loss

    def post_process(self, proposals, probs, minProb=0.2):
        proposalsFlat = proposals.reshape(-1, 7)
        proposalIdx = (probs > minProb).flatten()

        ### nms ###
        box3d_nms, nms_scores = net.rpn_nms_op.rpn_nms(probs.flatten()[proposalIdx], proposalsFlat[proposalIdx],
                                                       self.voxelnet.anchors[proposalIdx])
        return box3d_nms, nms_scores

    def save_weights(self, weights=None, dir=None):
        dir = self.ckpt_dir if dir == None else dir
        if weights == None:
            weights = [voxelnet.conv_net_name, voxelnet.rpn_name]
        for name in weights:
            if name == voxelnet.conv_net_name:
                self.subnet_conv.save_weights(self.sess, dir=os.path.join(dir, name))

            elif name == voxelnet.rpn_name:
                self.subnet_rpn.save_weights(self.sess, dir=os.path.join(dir, name))
            else:
                ValueError('unknow weigths name')

    def load_weights(self, weights=[]):
        for name in weights:
            if name == voxelnet.conv_net_name:
                self.subnet_conv.load_weights(self.sess)
            elif name == voxelnet.rpn_name:
                self.subnet_rpn.load_weights(self.sess)
            else:
                ValueError('unknow weigths name')

    def predict(self, top_view):
        fd1 = {
            self.placeholder['top_view']: top_view,
            net.layers.IS_TRAIN_PHASE: False
        }

        proposals, scores, probs, conv = self.sess.run( [self.placeholder['proposals'],
                                                         self.placeholder['scores'],
                                                         self.placeholder['probs'],
                                                         self.placeholder['conv']], fd1)
        return proposals, scores, probs, conv

    def summary_image(self, image, tag, summary_writer=None,step=None):

        #reduce image size
        #new_size = (image.shape[1]//2, image.shape[0]//2)
        #image=cv2.resize(image,new_size)

        if summary_writer == None:
            summary_writer=self.train_summary_writer

        im_summaries = []
        # Write the image to a string
        s = io.BytesIO()
        plt.imsave(s,image)

        # Create an Image object
        img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=image.shape[0],
                                   width=image.shape[1])
        # Create a Summary value
        im_summaries.append(tf.Summary.Value(tag=tag, image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        summary_writer.add_summary(summary, step)

    def log_image(self, step, frame_tag, prefix, summary_writer):
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.top_image = data.draw_top_image(self.batch_top_view[0])
        top_view_log = self.top_image.copy()
        # add text on origin
        text = frame_tag
        for i, line in enumerate(text.split("/")):
            y = 25 + i * 20
            cv2.putText(top_view_log, line, (5, y), font, 0.5, (0, 255, 100), 0, cv2.LINE_AA)

        #draw groundtruth on topview
        for gt_bbox in self.batch_gt_boxes3d[:]:
            topview = net.processing.boxes3d.draw_box3d_on_top(top_view_log, gt_bbox)
        self.summary_image(topview, prefix + '/top_view', step=step, summary_writer=summary_writer)