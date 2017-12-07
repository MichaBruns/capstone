import net.processing.boxes3d
import config as cfg
import numpy as np
import math

''''
Files for computing IOUs
'''

"""
Projects 3d bounding boxes onto topview image and calculates IOU for all anchors

Returns the anchors with the highest IOU
"""
def eval_anchors(gt_3dboxes, anchors):
    top_boxes = net.processing.boxes3d.box3d_to_top_box(gt_3dboxes)

    # project


"""
Create two anchors at each location with size (ANCHOR_LENGTH, ANCHOR_WIDTH, ANCHOR_HEIGHT) rotated by 0 and by 90 degree

Returns anchors in (midX, midY, midZ, length, width, height, theata) format and 3d Bounding boxes
"""
def create_anchors(step=1):

    anchors = []#np.zeros(())
    anchors3D = []

    for x in np.arange(cfg.TOP_X_MIN, cfg.TOP_X_MAX, cfg.TOP_X_DIVISION*step):
        for y in np.arange(cfg.TOP_Y_MIN, cfg.TOP_Y_MAX, cfg.TOP_Y_DIVISION*step):
            anchor = [x, y, cfg.ANCHOR_Z, cfg.ANCHOR_LENGTH, cfg.ANCHOR_WIDTH, cfg.ANCHOR_HEIGHT, 0]
            anchors.append(anchor)
            anchor = [x, y, cfg.ANCHOR_Z, cfg.ANCHOR_LENGTH, cfg.ANCHOR_WIDTH, cfg.ANCHOR_HEIGHT, math.radians(90)]
            anchors.append(anchor)

            anchorNP = np.asarray([ [x - cfg.ANCHOR_LENGTH/2, y - cfg.ANCHOR_WIDTH/2,
                                     cfg.ANCHOR_Z - cfg.ANCHOR_HEIGHT/2],
                                    [x - cfg.ANCHOR_LENGTH / 2, y + cfg.ANCHOR_WIDTH / 2,
                                     cfg.ANCHOR_Z - cfg.ANCHOR_HEIGHT / 2],
                                    [
                                        x + cfg.ANCHOR_LENGTH / 2, y + cfg.ANCHOR_WIDTH / 2,
                                        cfg.ANCHOR_Z - cfg.ANCHOR_HEIGHT / 2],
                                    [
                                        x + cfg.ANCHOR_LENGTH / 2, y - cfg.ANCHOR_WIDTH / 2,
                                        cfg.ANCHOR_Z - cfg.ANCHOR_HEIGHT / 2],
                                    [x - cfg.ANCHOR_LENGTH / 2, y - cfg.ANCHOR_WIDTH / 2,
                                     cfg.ANCHOR_Z + cfg.ANCHOR_HEIGHT / 2],
                                    [
                                        x - cfg.ANCHOR_LENGTH / 2, y + cfg.ANCHOR_WIDTH / 2,
                                        cfg.ANCHOR_Z + cfg.ANCHOR_HEIGHT / 2],
                                    [
                                        x + cfg.ANCHOR_LENGTH / 2, y + cfg.ANCHOR_WIDTH / 2,
                                        cfg.ANCHOR_Z + cfg.ANCHOR_HEIGHT / 2],
                                    [
                                        x + cfg.ANCHOR_LENGTH / 2, y - cfg.ANCHOR_WIDTH / 2,
                                        cfg.ANCHOR_Z + cfg.ANCHOR_HEIGHT / 2]])
            anchors3D.append(anchorNP)

            anchorNP = np.asarray(
                [[x - cfg.ANCHOR_WIDTH / 2, y - cfg.ANCHOR_LENGTH / 2,
                  cfg.ANCHOR_Z - cfg.ANCHOR_HEIGHT / 2],
                 [x - cfg.ANCHOR_WIDTH / 2, y + cfg.ANCHOR_LENGTH / 2,
                  cfg.ANCHOR_Z - cfg.ANCHOR_HEIGHT / 2],
                 [
                     x + cfg.ANCHOR_WIDTH / 2, y + cfg.ANCHOR_LENGTH / 2,
                     cfg.ANCHOR_Z - cfg.ANCHOR_HEIGHT / 2],
                 [
                     x + cfg.ANCHOR_WIDTH / 2, y - cfg.ANCHOR_LENGTH / 2,
                     cfg.ANCHOR_Z - cfg.ANCHOR_HEIGHT / 2],
                 [x - cfg.ANCHOR_WIDTH / 2, y - cfg.ANCHOR_LENGTH / 2,
                  cfg.ANCHOR_Z + cfg.ANCHOR_HEIGHT / 2],
                 [
                     x - cfg.ANCHOR_WIDTH / 2, y + cfg.ANCHOR_LENGTH / 2, cfg.ANCHOR_Z + cfg.ANCHOR_HEIGHT / 2],
                 [
                     x + cfg.ANCHOR_WIDTH / 2, y + cfg.ANCHOR_LENGTH / 2, cfg.ANCHOR_Z + cfg.ANCHOR_HEIGHT / 2],
                 [
                     x + cfg.ANCHOR_WIDTH / 2, y - cfg.ANCHOR_LENGTH / 2, cfg.ANCHOR_Z + cfg.ANCHOR_HEIGHT / 2]])
            anchors3D.append(anchorNP)

    return np.asarray(anchors), anchors3D