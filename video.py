import logging

import numpy as np
import cv2
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)

def read_all_frames(cv2_cap: cv2.VideoCapture, segment=None):
    """read all frames of a cv2 VideoCapture into 3D numpy array"""
    varr = []
    ii = -1
    while cv2_cap.isOpened():
        ii += 1
        if segment is not None and ii<segment[0]:
            continue
        elif segment is not None and ii>segment[1]:
            break
        status, frame = cv2_cap.read()
        if not status:
            break
        varr.append(frame)
    varr = np.array(varr)
    return varr

def resample_all_frames(varr, zfactor):
    if not isinstance(zfactor, (tuple, list)):
        # don't scale color channel axis
        zfactor = (zfactor, zfactor, 1.0)
    if zfactor[0]==1.0 and zfactor[1]==1.0:
        hh, ww = varr[0].shape[:2]
        return varr, ww, hh

    revarr = []
    for f in varr:
        revarr.append( zoom(f, zfactor) )
    hh, ww = revarr[0].shape[:2]
    return np.array(revarr), ww, hh

def grayscale_all_frames(varr):
    revarr = []
    for f in varr:
        revarr.append( cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) )
    return np.array(revarr)

def ftou8(frame):
    return np.uint8((frame-np.min(frame))/np.max(frame)*255)

