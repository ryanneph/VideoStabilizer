import logging
import math

import numpy as np
import cv2

from video import ftou8

logger = logging.getLogger(__name__)

def calculate_features(vg):
    # calculate tracking points
    pts = []
    for f in vg:
        f = cv2.GaussianBlur(f, (5,5), 0.5)
        pts.append( cv2.goodFeaturesToTrack(f, maxCorners=30, qualityLevel=0.05, minDistance=30, blockSize=3) )
    return np.squeeze(np.array(pts))

def _calc_optical_flow(frame1, points1, frame2):
    """Calculate sparse (feature-based) optical flow between two frames"""
    estim_pts, status, err = cv2.calcOpticalFlowPyrLK(ftou8(frame1), ftou8(frame2), points1, None)
    # only use valid points
    filter = np.where(status==1)[0]
    prev_pts = points1[filter]
    curr_pts = estim_pts[filter]
    return prev_pts, curr_pts

def _calculate_motion(vg, pts, calcfunc):
    """optical flow on feature points and camera transform estimation using modular function 'calcfunc' """
    flow = []
    for ii in range(1,len(vg)):
        prev_pts, curr_pts = _calc_optical_flow(vg[ii-1], pts[ii-1], vg[ii])
        matrix, in_mask = calcfunc(prev_pts, curr_pts)
        flow.append(matrix)
    return np.squeeze(np.array(flow))

def calculate_motion_perspective(vg, pts):
    return _calculate_motion(vg, pts, lambda a,b: cv2.findHomography(a,b,cv2.RANSAC))

def calculate_motion_affine(vg, pts):
    return _calculate_motion(vg, pts, lambda a,b: cv2.estimateAffine2D(a,b))

def decompose_affine(As, vectors=False):
    """decompose affine matrix into (scale, rotation, translation)"""
    if As.ndim == 2:
        As = As[np.newaxis, :]
    m, n = As.shape[1:]

    Sout = []
    Rout = []
    Tout = []
    for A in As:
        T = np.eye(n)
        T[:,-1] = np.concatenate((A[:,-1], [1]))
        S = np.diag([np.linalg.norm(A[:,ii]) for ii in range(n-1)]+[1])
        R = np.divide(np.concatenate((A, np.zeros((1,n))), axis=0), np.diagonal(S).T)
        R[:,-1] = 0
        R[-1, -1] = 1

        if vectors:
            Sout.append(np.diagonal(S)[:-1])
            Rout.append(np.arctan2(R[1,0], R[0,0]))
            Tout.append(T[:-1, -1])
        else:
            Sout.append(S)
            Rout.append(R)
            Tout.append(T)
    return (np.array(Sout), np.array(Rout), np.array(Tout))

def compose_affine(sx, sy, r, tx, ty):
    N = len(sx)
    cosr, sinr = np.cos(r), np.sin(r)
    A = np.zeros((N, 2, 3))
    A[:,  0, -1] = tx
    A[:,  1, -1] = ty
    A[:,  0,  0] = sx*cosr
    A[:,  0,  1] = -sx*sinr
    A[:,  1,  0] = sy*sinr
    A[:,  1,  1] = sy*cosr
    return A

