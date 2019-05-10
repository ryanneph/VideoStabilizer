import sys, os
from os.path import join as pjoin
import argparse
import logging
import time
from pprint import pprint

import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('infile', metavar='in', help='input video file')
parser.add_argument('-o', '--out', help='input video file')
parser.add_argument('-r', '--re', '--resample', type=float, dest='resample', default=1.0, help='resample with zoom factor')
parser.add_argument('-L', '--loglevel', default='DEBUG', choices=list(logging._nameToLevel.keys()), help='set logging level')
args = parser.parse_args()
loglevel = logging._nameToLevel[args.loglevel]

logger = logging.getLogger(__name__)
logger.addHandler( logging.StreamHandler(sys.stdout) )
logging.getLogger().setLevel(loglevel)


def read_all_frames(cv2_cap: cv2.VideoCapture):
    """read all frames of a cv2 VideoCapture into 3D numpy array"""
    varr = []
    while cv2_cap.isOpened():
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
        return varr

    revarr = []
    for f in varr:
        revarr.append( zoom(f, zfactor) )
    return np.array(revarr)

def grayscale_all_frames(varr):
    revarr = []
    for f in varr:
        revarr.append( cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) )
    return np.array(revarr)

def calculate_features(vg):
    # calculate tracking points
    pts    = []
    for f in vg:
        f = cv2.GaussianBlur(f, (5,5), 0.2)
        pts.append( cv2.goodFeaturesToTrack(f, maxCorners=20, qualityLevel=0.01, minDistance=30, blockSize=3) )
    return np.squeeze(np.array(pts))

def calculate_camera_motion(features):
    # convert to grayscale
    vg = grayscale_all_frames(varr)
    #  curr_pts, status, err = cv2.calcOpticalFlowPyrLK(vg[ii], vg[ii+1], )
    return None

def interactive_play_video(varr, framerate=None, features=None):
    if not framerate:
        framerate = 24
    playing = True

    ii = 0
    while ii<varr.shape[0]:
        frame = varr[ii]
        pts = features[ii]
        for pt in pts:
            cv2.circle(frame, (pt[0], pt[1]), 5, (0,0,255), -1)
        cv2.imshow('', frame)
        while True:
            keyp = cv2.waitKey(int(1/framerate*1000))

            if keyp==ord(' ') or keyp==83: #right
                break
            elif keyp==81: # left
                ii-=2
                break
            elif keyp==ord('p'):
                playing = not playing
                break
            elif keyp==ord('q') :
                sys.exit(0)
            elif playing:
                break
        if ii == varr.shape[0]-1:
            ii = 0
            continue
        ii += 1


if __name__ == '__main__':
    vs = cv2.VideoCapture(args.infile)
    nframes = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    hh      = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ww      = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    frate   = int(vs.get(cv2.CAP_PROP_FPS))

    logger.debug('Read video file:\n' + \
                 '  Frames:     {}\n'.format(nframes) + \
                 '  Size (HxW): {}x{}\n'.format(hh, ww) + \
                 '  Rate (fps): {}\n'.format(frate) )

    varr = read_all_frames(vs)
    vz = resample_all_frames(varr, args.resample)
    vg = grayscale_all_frames(varr)

    features = calculate_features(vg)
    camera_motion = calculate_camera_motion(features)

    #  plt.imshow(vg[0], cmap='gray')
    #  print(features.shape, features[0])
    #  plt.scatter(features[0, :, 0], features[0, :, 1], marker='+', color='red')
    #  plt.show()

    interactive_play_video(vz, features=features, framerate=frate)
