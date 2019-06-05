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
parser.add_argument('-o', '--out', default=None, help='input video file')
parser.add_argument('-r', '--re', '--resample', type=float, dest='resample', default=1.0, help='resample with zoom factor')
parser.add_argument('--vis', '--visualize', dest='visualize', action='store_true', help='play video with feature annotation')
parser.add_argument('-L', '--loglevel', default='DEBUG', choices=list(logging._nameToLevel.keys()), help='set logging level')
parser.add_argument('--fseg', '--frame-segment', type=int, dest='fsegment', nargs=2, default=None, help='start and end frames for processing')
parser.add_argument('--tseg', '--time-segment', type=float, dest='tsegment', nargs=2, default=None, help='start and end times for processing')
args = parser.parse_args()
loglevel = logging._nameToLevel[args.loglevel]

logger = logging.getLogger(__name__)
logger.addHandler( logging.StreamHandler(sys.stdout) )
logging.getLogger().setLevel(loglevel)

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
        f = cv2.GaussianBlur(f, (5,5), 0.5)
        pts.append( cv2.goodFeaturesToTrack(f, maxCorners=40, qualityLevel=0.03, minDistance=30, blockSize=3) )
    return np.squeeze(np.array(pts))

def ftou8(frame):
    return np.uint8((frame-np.min(frame))/np.max(frame)*255)

def calculate_feature_flow(vg, pts):
    """optical flow on feature points"""
    flow = []
    for ii in range(1,len(vg)):
        estim_pts, status, err = cv2.calcOpticalFlowPyrLK(ftou8(vg[ii-1]), ftou8(vg[ii]), pts[ii-1], None)
        # only use valid points
        filter = np.where(status==1)[0]
        prev_pts = pts[ii-1][filter]
        curr_pts = estim_pts[filter]
        flow.append( cv2.findHomography(prev_pts, curr_pts) )
        #  matcher=cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        #  knn_matches = matcher.knnMatch(prev_pts, curr_pts, 2)
    return np.squeeze(np.array(flow))

def interactive_play_video(varr, framerate=None, features=None):
    if not framerate:
        framerate = 24
    playing = True

    ii = 0
    while ii<varr.shape[0]:
        frame = varr[ii].copy()
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
                return
            elif playing:
                break
        if ii == varr.shape[0]-1:
            ii = 0
            continue
        ii += 1


if __name__ == '__main__':
    time_start = time.perf_counter()

    vs = cv2.VideoCapture(args.infile)
    nframes = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    hh      = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ww      = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    frate   = int(vs.get(cv2.CAP_PROP_FPS))

    logger.debug('Video Metadata:\n' + \
                 '  Frames:     {}\n'.format(nframes) + \
                 '  Size (HxW): {}x{}\n'.format(hh, ww) + \
                 '  Rate (fps): {}'.format(frate) )

    segment = None
    if args.fsegment or args.tsegment:
        if args.tsegment:
            segment = ( int(args.tsegment[0]//(1/frate)), int(args.tsegment[1]//(1/frate)) )
        else:
            segment = fsegment
        segment = (max(0, segment[0]), min(nframes, segment[1]))
        logger.debug('  Segment:    {}-{}'.format(*segment))
    logger.debug('')

    logger.info('Reading video frames...')
    varr = read_all_frames(vs, segment=segment)
    vz = resample_all_frames(varr, args.resample)
    vg = grayscale_all_frames(varr)

    logger.info('Calculating tracking features...')
    features = calculate_features(vg) # shape: [nframes, npoints, dims=2]
    logger.info('Calculating camera motion...')
    camera_motion = calculate_feature_flow(vg, features)
    pprint(camera_motion)

    if args.visualize:
        #  plt.imshow(vg[0], cmap='gray')
        #  print(features.shape, features[0])
        #  plt.scatter(features[0, :, 0], features[0, :, 1], marker='+', color='red')
        #  plt.show()
        interactive_play_video(vz, features=features, framerate=frate)

    # save video out
    logger.info('Saving stabilized footage...')
    if args.out is None:
        outfile = 'stabilized.mp4'
    vout = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'mp4v'), frate, (ww, hh))
    for f in vz:
        vout.write(f)
    logger.info('Stabilized footage saved to "{}"'.format(outfile))
    logger.debug('Total runtime: {:0.2f}s'.format(time.perf_counter()-time_start))
