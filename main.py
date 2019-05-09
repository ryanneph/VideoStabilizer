import sys, os
from os.path import join as pjoin
import argparse
import logging
import time

import numpy as np
from scipy.ndimage import zoom
#import matplotlib.pyplot as plt
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

def interactive_play_video(varr):
    ii = 0
    while ii<varr.shape[0]:
        cv2.imshow('', varr[ii])
        while True:
            keyp = cv2.waitKey()
            time.sleep(1/30)
            if keyp==ord(' '):
                break
            if keyp==ord('q'):
                sys.exit(0)
        if ii == varr.shape[0]-1:
            ii = 0
            continue
        ii += 1


if __name__ == '__main__':
    vs = cv2.VideoCapture(args.infile)
    nframes = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    hh      = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ww      = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))

    logger.debug('Read video file:\n' + \
                 '  Frames:     {}\n'.format(nframes) + \
                 '  Size (HxW): {}x{}\n'.format(hh, ww))

    varr = read_all_frames(vs)
    vz = resample_all_frames(varr, args.resample)

    interactive_play_video(vz)
