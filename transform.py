import numpy as np
import cv2

def _warp_sequence(varr, mat, warpfunc):
    vout = []
    for ii in range(len(varr)-1):
        inim = varr[ii]
        outim = warpfunc(inim, mat[ii], inim.shape[:2][::-1])
        vout.append(np.array(outim))
    return np.squeeze(np.array(vout))

def warp_sequence_perspective(varr, mat):
    return _warp_sequence(varr, mat, lambda a,b,c: cv2.warpPerspective(a,b,c))

def warp_sequence_affine(varr, mat):
    return _warp_sequence(varr, mat, lambda a,b,c: cv2.warpAffine(a,b,c))

def transform_points(points, mats):
    newpoints = []
    for ii in range(len(mats)):
        newpoints.append(cv2.transform(np.array([points[ii]]), mats[ii]))
    return np.squeeze(np.array(newpoints))
