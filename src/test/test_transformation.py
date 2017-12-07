import numpy as np
import matplotlib
import utils.boxes
import net.rpn_target_op
from shapely.geometry import Polygon
import net.processing.boxes
import math
import matplotlib.pyplot as plt
import matplotlib.rcsetup as rcsetup
print(matplotlib.get_backend())
"""
Unit test for 3d->voxel transformation
"""

showDiff = False
"""
Rotate point around Z axis
"""
def rotate_point(theta, center, dimensions):
    c, s = np.cos(theta), np.sin(theta)
    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    R[2, 2] = 1
    displace = np.dot(dimensions, R.transpose())
    return center + displace

def transform_boundingBoxes(boundingBoxes, anchors, anchors_top, validIDs):
    bb_top = net.processing.boxes3d.box3d_to_top_box(boundingBoxes)

    pos_neg_inds, pos_inds, labels, targets, argmax_overlaps, idx_target = net.rpn_target_op.rpn_target(anchors_top,
                                                                                                        validIDs, None,
                                                                                                        bb_top,
                                                                                                        anchors,
                                                                                                        boundingBoxes)
    pos_anchors = np.asarray(anchors)[idx_target]
    return targets, pos_anchors

def get_poly_difference(bbox1, bbox2):
    poly1 = Polygon(bbox1)
    poly2 = Polygon(bbox2)
    diff = poly1.symmetric_difference(poly2)
    plt.switch_backend("TkAgg")
    if diff.area > 0.01 and showDiff:
        poly1points = np.array(list(poly1.exterior.coords))
        poly2points = np.array(list(poly2.exterior.coords))
        plt.plot(poly1points[:,0], poly1points[:,1], '--')
        plt.plot(poly2points[:, 0], poly2points[:, 1], '-.')
        for poly in diff:
            coords = poly.exterior.coords.xy
            plt.plot(coords[0], coords[1], '-')

        plt.show()
    return diff.area

def test_theta(theta, center, dimensions, anchors, anchors_top, validIDs):
    bbox = rotate_point(theta,center, dimensions)
    boundingBoxes = []
    boundingBoxes.append(bbox)
    boundingBoxes = np.asarray(boundingBoxes)

    targets, pos_anchors = transform_boundingBoxes(boundingBoxes, anchors, anchors_top, validIDs)
    if ( abs(abs(theta)-math.radians(90)) < 0.001):
        # special treatment for angles near 90 deg
        return (np.all( (np.abs((targets[:, -1] + pos_anchors[:, -1]) - theta) < 0.001) or (np.abs((targets[:, -1] + pos_anchors[:, -1]) + theta) < 0.001)))
    else:
        return (np.all(np.abs((targets[:, -1] + pos_anchors[:, -1]) - theta) < 0.001))


def test_reproject(theta, center, dimensions, anchors, anchors_top, validIDs):

    bbox = rotate_point(theta,center, dimensions)
    boundingBoxes = []
    boundingBoxes.append(bbox)
    boundingBoxes = np.asarray(boundingBoxes)
    targets, pos_anchors = transform_boundingBoxes(boundingBoxes, anchors, anchors_top, validIDs)
    reprojectedBoxes = net.processing.boxes.box_transform_voxelnet_inv(targets, pos_anchors)
    diff = get_poly_difference(bbox[:4, :2], reprojectedBoxes[0,:4, :2])
    return (diff < 0.01)

anchors, anchors3d = utils.boxes.create_anchors()
aa = anchors.reshape( (176,200, 7 ))
anchors_top = net.processing.boxes3d.box3d_to_top_box(anchors3d)
validIds =  np.arange(0, len(anchors))

#create 3d bounding boxes with shape 8x3
width = 3
length = 4
height = 2
center = np.asarray( [ 60, 30, 1], dtype=np.float32 )

bbox = np.zeros( (8,3), dtype=np.float32)
bbox[0] = center + [-length/2, -width/2, -height/2]
bbox[1] = center + [-length/2,  width/2, -height/2]
bbox[2] = center + [ length/2,  width/2, -height/2]
bbox[3] = center + [ length/2, -width/2, -height/2]
bbox[4] = center + [-length/2, -width/2,  height/2]
bbox[5] = center + [-length/2,  width/2,  height/2]
bbox[6] = center + [ length/2,  width/2,  height/2]
bbox[7] = center + [ length/2, -width/2,  height/2]

dimensions = np.zeros( (8, 3), dtype=np.float32)
dimensions[0] = [-length/2, -width/2, -height/2]
dimensions[1] = [-length/2,  width/2, -height/2]
dimensions[2] = [ length/2,  width/2, -height/2]
dimensions[3] = [ length/2, -width/2, -height/2]
dimensions[4] = [-length/2, -width/2,  height/2]
dimensions[5] = [-length/2,  width/2,  height/2]
dimensions[6] = [ length/2,  width/2,  height/2]
dimensions[7] = [ length/2, -width/2,  height/2]

assert(test_theta(math.radians(0), center, dimensions, anchors, anchors_top, validIds))
assert(test_theta(math.radians(90), center, dimensions, anchors, anchors_top, validIds))
assert(test_theta(math.radians(90), center, dimensions, anchors, anchors_top, validIds))
assert(test_theta(math.radians(45), center, dimensions, anchors, anchors_top, validIds))
assert(test_theta(math.radians(-45), center, dimensions, anchors, anchors_top, validIds))
assert(test_theta(math.radians(30), center, dimensions, anchors, anchors_top, validIds))
assert(test_theta(math.radians(-30), center, dimensions, anchors, anchors_top, validIds))
assert(test_theta(math.radians(15), center, dimensions, anchors, anchors_top, validIds))
assert(test_theta(math.radians(-15), center, dimensions, anchors, anchors_top, validIds))

bbox1 = rotate_point(0, center, dimensions)

# test with changed dimensions
dimensions = np.zeros( (8, 3), dtype=np.float32)
dimensions[0] = [ length/2, -width/2, -height/2]
dimensions[1] = [-length/2, -width/2, -height/2]
dimensions[2] = [-length/2,  width/2, -height/2]
dimensions[3] = [ length/2,  width/2, -height/2]
dimensions[4] = [ length/2, -width/2,  height/2]
dimensions[5] = [-length/2, -width/2,  height/2]
dimensions[6] = [-length/2,  width/2,  height/2]
dimensions[7] = [ length/2,  width/2,  height/2]

assert(test_theta(math.radians(0), center, dimensions, anchors, anchors_top, validIds))
assert(test_theta(math.radians(90), center, dimensions, anchors, anchors_top, validIds))
assert(test_theta(math.radians(90), center, dimensions, anchors, anchors_top, validIds))
assert(test_theta(math.radians(45), center, dimensions, anchors, anchors_top, validIds))
assert(test_theta(math.radians(-45), center, dimensions, anchors, anchors_top, validIds))
assert(test_theta(math.radians(30), center, dimensions, anchors, anchors_top, validIds))
assert(test_theta(math.radians(-30), center, dimensions, anchors, anchors_top, validIds))
assert(test_theta(math.radians(15), center, dimensions, anchors, anchors_top, validIds))
assert(test_theta(math.radians(-15), center, dimensions, anchors, anchors_top, validIds))


'''
bbox2 = rotate_point(0, center, dimensions)

Poly1 = Polygon(bbox1[:4,:2])
Poly2 = Polygon(bbox2[:4,:2])
intersection = Poly1.symmetric_difference(Poly2)
a =  np.asarray(list(Poly1.exterior.coords))
b = np.asarray(list(Poly2.exterior.coords))
c = np.asarray(list(intersection.exterior.coords))
plt.switch_backend("TkAgg")
plt.plot(a[:,0], a[:,1],'--')
plt.plot(b[:,0], b[:,1],':')
plt.plot(c[:,0], c[:,1],'-.')
plt.show()
'''

#reprojection tests
assert(test_reproject(math.radians(0), center, dimensions, anchors, anchors_top, validIds))
assert (test_reproject(math.radians(90), center, dimensions, anchors, anchors_top, validIds))
assert (test_reproject(math.radians(-90), center, dimensions, anchors, anchors_top, validIds))
assert (test_reproject(math.radians(45), center, dimensions, anchors, anchors_top, validIds))
assert (test_reproject(math.radians(-45), center, dimensions, anchors, anchors_top, validIds))
assert (test_reproject(math.radians(30), center, dimensions, anchors, anchors_top, validIds))
assert (test_reproject(math.radians(-30), center, dimensions, anchors, anchors_top, validIds))
assert (test_reproject(math.radians(15), center, dimensions, anchors, anchors_top, validIds))
assert (test_reproject(math.radians(-15), center, dimensions, anchors, anchors_top, validIds))

gg = True