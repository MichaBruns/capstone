from net.configuration import CFG
from net.lib.utils.bbox import bbox_overlaps ,box_vote
from net.lib.nms.py_cpu_nms import py_cpu_nms as nms
import numpy as np
import math

#     roi  : i, x1,y1,x2,y2  i=image_index  
#     box : x1,y1,x2,y2,


def clip_boxes(boxes, width, height):
    ''' Clip process to image boundaries. '''

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], width - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], height - 1), 0)
    # x2 < width
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], width - 1), 0)
    # y2 < height
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], height - 1), 0)
    return boxes


# et_boxes = estimated bounding boxes in voxel format
# gt_boxes = ground truth in 3d
def box_transform_voxelnet(et_boxes, gt_boxes):
    et_ws = et_boxes[:, 4]
    et_ls = et_boxes[:, 3]
    et_hs = et_boxes[:, 5]
    et_cxs = et_boxes[:, 0]
    et_cys = et_boxes[:, 1]
    et_czs = et_boxes[:, 2]
    et_theta = et_boxes[:, -1]
    et_d = np.sqrt(et_ws*et_ws + et_ls*et_ls)

    # assume that length > width
    t1 = np.linalg.norm(gt_boxes[:,1,:2] - gt_boxes[:,0,:2], axis=1)
    #gt_dist2 = np.linalg.norm(gt_boxes[:, 2, :2] - gt_boxes[:, 0, :2], axis=1)
    t2 = np.linalg.norm(gt_boxes[:, 3, :2] - gt_boxes[:, 0, :2], axis=1)


    gt_ls = np.maximum(t1, t2)
    gt_ws = np.minimum(t1, t2)
    gt_hs = np.abs(gt_boxes[:, 4, 2] - gt_boxes[:, 0, 2])

    boxSum = np.sum(gt_boxes, axis=(1))
    gt_cxs = boxSum[:, 0] / 8
    gt_cys = boxSum[:, 1] / 8
    gt_czs = boxSum[:, 2] / 8


    # extract the theta
    # theta should be in range (+pi/2, -pi/2)
    # therefore diffX must be positive

    assert(all(gt_ls > gt_ws))

    diffX = gt_boxes[:, 3, 0] - gt_boxes[:,0, 0]
    diffX2 = gt_boxes[:, 1, 0] - gt_boxes[:, 0, 0]
    diffX = np.where(t1 > t2,  diffX2, diffX)
    inversed = np.where(diffX < 0)
    diffX[inversed] *= -1
    diffY = gt_boxes[:, 3, 1] - gt_boxes[:, 0, 1]
    diffY2 = gt_boxes[:, 1, 1] - gt_boxes[:, 0, 1]
    diffY = np.where(t1 > t2,  diffY2, diffY)
    diffY[inversed] *= -1

    gt_theta = np.arctan2(diffY, diffX)

    assert (all(gt_theta >= - math.radians(90)))
    assert(all(gt_theta <= math.radians(90)))

    dxs = (gt_cxs - et_cxs) / et_d
    dys = (gt_cys - et_cys) / et_d
    dzs = (gt_czs - et_czs) / et_hs
    dws = np.log(gt_ws / et_ws)
    dls = np.log(gt_ls / et_ls)
    dhs = np.log(gt_hs / et_hs)
    dtheta = gt_theta - et_theta

    deltas = np.vstack((dxs, dys, dzs, dls, dws, dhs, dtheta)).transpose()
    return deltas

'''
Transforms voxelnet representation (dX, dY, dZ, dl, dW, dH, dTheta) to 3d bounding boxes
'''
def box_transform_voxelnet_inv(deltas, anchors):

    num = len(deltas)
    boxes = np.zeros((num,8,3), dtype=np.float32)
    if num == 0: return boxes

    a_cxs = anchors[:, 0]
    a_cys = anchors[:, 1]
    a_czs = anchors[:, 2]
    a_ws = anchors[:, 4]
    a_ls = anchors[:, 3]
    a_hs = anchors[:, 5]
    a_d = np.sqrt(a_ws*a_ws + a_ls*a_ls)

    mid= np.zeros((num, 3), dtype=np.float32)
    mid[:, 0] = a_d * deltas[:,0] + a_cxs
    mid[:, 1] = a_d * deltas[:, 1] + a_cys
    mid[:, 2] = a_hs * deltas[:, 2] + a_czs
    cxs = a_d * deltas[:,0] + a_cxs
    cys = a_d * deltas[:,1] + a_cys
    czs = a_hs * deltas[:,2] + a_czs
    length =np.exp(deltas[:,3]) * anchors[:,3]
    width = np.exp(deltas[:,4]) * anchors[:,4]
    height = np.exp(deltas[:,5]) * anchors[:,5]
    theta = deltas[:, 6] + anchors[:,6]

#  there must be a better way..

    relPos = np.zeros((num, 8, 3), dtype=np.float32)
    relPos[:, 0, 0] = - length/2
    relPos[:, 1, 0] = + length / 2
    relPos[:, 2, 0] = + length / 2
    relPos[:, 3, 0] = - length / 2

    relPos[:, 4, 0] = - length / 2
    relPos[:, 5, 0] = + length / 2
    relPos[:, 6, 0] = + length / 2
    relPos[:, 7, 0] = - length / 2

    relPos[:, 0, 1] = - width / 2
    relPos[:, 1, 1] = - width / 2
    relPos[:, 2, 1] = + width / 2
    relPos[:, 3, 1] = + width / 2

    relPos[:, 4, 1] = - width / 2
    relPos[:, 5, 1] = - width / 2
    relPos[:, 6, 1] = + width / 2
    relPos[:, 7, 1] = + width / 2

    relPos[:, 0, 2] = -height/2
    relPos[:, 1, 2] = -height / 2
    relPos[:, 2, 2] = -height / 2
    relPos[:, 3, 2] = -height / 2

    relPos[:, 4, 2] = height / 2
    relPos[:, 5, 2] = height / 2
    relPos[:, 6, 2] = height / 2
    relPos[:, 7, 2] = height / 2




    # create rotation matrix
    c, s = np.cos(theta), np.sin(theta)
    R= np.zeros( (num, 3, 3), dtype=np.float32)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1


    #R = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotRel = np.zeros((num, 8, 3), dtype=np.float32)
    for i in range(num):
        rotRel[i] = np.dot(relPos[i], R[i].transpose())
    #rotRel = np.tensordot(relPos, R)
    boxes = rotRel + np.expand_dims(mid, axis=1)
    '''
    boxes[:, 0, 0] = cxs - (length/2)
    boxes[:, 1, 0] = cxs + (length/2)
    boxes[:, 2, 0] = cxs + (length/2)
    boxes[:, 3, 0] = cxs - (length/2)
    boxes[:, 4, 0] = cxs - (length/2)
    boxes[:, 5, 0] = cxs + (length/2)
    boxes[:, 6, 0] = cxs + (length/2)
    boxes[:, 7, 0] = cxs - (length/2)

    boxes[:, 0, 1] = cys - (width / 2)
    boxes[:, 1, 1] = cys - (width / 2)
    boxes[:, 2, 1] = cys + (width / 2)
    boxes[:, 3, 1] = cys + (width / 2)

    boxes[:, 4, 1] = cys - (width / 2)
    boxes[:, 5, 1] = cys - (width / 2)
    boxes[:, 6, 1] = cys + (width / 2)
    boxes[:, 7, 1] = cys + (width / 2)

    boxes[:, 0, 2] = czs - (height / 2)
    boxes[:, 1, 2] = czs - (height / 2)
    boxes[:, 2, 2] = czs - (height / 2)
    boxes[:, 3, 2] = czs - (height / 2)

    boxes[:, 4, 2] = czs + (height / 2)
    boxes[:, 5, 2] = czs + (height / 2)
    boxes[:, 6, 2] = czs + (height / 2)
    boxes[:, 7, 2] = czs + (height / 2)
    '''

    return boxes

def box_transform(et_boxes, gt_boxes):
    et_ws  = et_boxes[:, 2] - et_boxes[:, 0] + 1.0
    et_hs  = et_boxes[:, 3] - et_boxes[:, 1] + 1.0
    et_cxs = et_boxes[:, 0] + 0.5 * et_ws
    et_cys = et_boxes[:, 1] + 0.5 * et_hs
     
    gt_ws  = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_hs  = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_cxs = gt_boxes[:, 0] + 0.5 * gt_ws
    gt_cys = gt_boxes[:, 1] + 0.5 * gt_hs
     
    dxs = (gt_cxs - et_cxs) / et_ws
    dys = (gt_cys - et_cys) / et_hs
    dws = np.log(gt_ws / et_ws)
    dhs = np.log(gt_hs / et_hs)

    deltas = np.vstack((dxs, dys, dws, dhs)).transpose()
    return deltas



def box_transform_inv(et_boxes, deltas):

    num = len(et_boxes)
    boxes = np.zeros((num,4), dtype=np.float32)
    if num == 0: return boxes

    et_ws  = et_boxes[:, 2] - et_boxes[:, 0] + 1.0
    et_hs  = et_boxes[:, 3] - et_boxes[:, 1] + 1.0
    et_cxs = et_boxes[:, 0] + 0.5 * et_ws
    et_cys = et_boxes[:, 1] + 0.5 * et_hs

    et_ws  = et_ws [:, np.newaxis]
    et_hs  = et_hs [:, np.newaxis]
    et_cxs = et_cxs[:, np.newaxis]
    et_cys = et_cys[:, np.newaxis]

    dxs = deltas[:, 0::4]
    dys = deltas[:, 1::4]
    dws = deltas[:, 2::4]
    dhs = deltas[:, 3::4]

    cxs = dxs * et_ws + et_cxs
    cys = dys * et_hs + et_cys
    # print('value for et_ws: ', et_ws)
    # print('value for dws: ', dws)
    ws  = np.exp(dws) * et_ws
    hs  = np.exp(dhs) * et_hs
    # print('ws is here: ', ws)
    # print('hs is here: ', hs)

    boxes[:, 0::4] = cxs - 0.5 * ws  # x1, y1,x2,y2
    boxes[:, 1::4] = cys - 0.5 * hs
    boxes[:, 2::4] = cxs + 0.5 * ws
    boxes[:, 3::4] = cys + 0.5 * hs

    return boxes

# nms  ###################################################################
def non_max_suppress(boxes, scores, num_classes,
                     nms_after_thesh=CFG.TEST.RCNN_NMS_AFTER, 
                     nms_before_score_thesh=0.05, 
                     is_box_vote=False,
                     max_per_image=100 ):

   
    # nms_before_thesh = 0.05 ##0.05   # set low number to make roc curve.
                                       # else set high number for faster speed at inference
 
    #non-max suppression 
    nms_boxes = [[]for _ in range(num_classes)]
    for j in range(1, num_classes): #skip background
        inds = np.where(scores[:, j] > nms_before_score_thesh)[0]
         
        cls_scores = scores[inds, j]
        cls_boxes  = boxes [inds, j*4:(j+1)*4]
        cls_dets   = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False) 

        # is_box_vote=0
        if len(inds)>0:
            keep = nms(cls_dets, nms_after_thesh) 
            dets_NMSed = cls_dets[keep, :] 
            if is_box_vote:
                cls_dets = box_vote(dets_NMSed, cls_dets)
            else:
                cls_dets = dets_NMSed 

        nms_boxes[j] = cls_dets
      

    ##Limit to MAX_PER_IMAGE detections over all classes
    if max_per_image > 0:
        image_scores = np.hstack([nms_boxes[j][:, -1] for j in range(1, num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in range(1, num_classes):
                keep = np.where(nms_boxes[j][:, -1] >= image_thresh)[0]
                nms_boxes[j] = nms_boxes[j][keep, :]

    return nms_boxes  
