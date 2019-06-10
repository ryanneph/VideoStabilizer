import sys, os
from os.path import join as pjoin
import argparse
import logging
import time
from pprint import pprint

import numpy as np
import cv2

import video
import sfm
import stabilize
import transform
import visualize

parser = argparse.ArgumentParser()
parser.add_argument('infile', metavar='in', help='input video file')
parser.add_argument('-o', '--out', default=None, help='input video file')
parser.add_argument('--transform', default='affine', choices=['affine', 'perspective'], help='camera motion model')
parser.add_argument('--radius', type=int, default=3, help='moving average filter radius')
parser.add_argument('--resample', '--re', type=float, dest='resample', default=1.0, help='resample with zoom factor')
parser.add_argument('--vis', '--visualize', dest='visualize', action='store_true', help='play video with feature annotation')
parser.add_argument('-L', '--loglevel', default='DEBUG', choices=list(logging._nameToLevel.keys()), help='set logging level')
parser.add_argument('--fseg', '--frame-segment', type=int, dest='fsegment', nargs=2, default=None, help='start and end frames for processing')
parser.add_argument('--tseg', '--time-segment', type=float, dest='tsegment', nargs=2, default=None, help='start and end times for processing')
parser.add_argument('--nostabilize', '--nostab', action='store_false', dest='stabilize', default=True, help='skip stabilization, only resample')
args = parser.parse_args()
loglevel = logging._nameToLevel[args.loglevel]

logger = logging.getLogger(__name__)
logger.addHandler( logging.StreamHandler(sys.stdout) )
logging.getLogger().setLevel(loglevel)

def stabilize_perspective(vz, vg, points):
    Mseq= sfm.calculate_motion_perspective(vg, points)
    trajectory = np.cumsum(Mseq, axis=0)
    sm_trajectory = stabilize.smooth_motion(trajectory, radius=args.radius)
    vstab = transform.warp_sequence_perspective(vz, Mseq+(sm_trajectory-trajectory))

    if args.visualize:
        visualize.plot_motion(trajectory, sm_trajectory)
        visualize.interactive_play_video(vz, features=points, framerate=frate)
        visualize.interactive_play_video(vstab, framerate=frate)

    return vstab, sm_trajectory

def stabilize_affine(vz, vg, points):
    Aseq = sfm.calculate_motion_affine(vg, points)

    S, R, T = sfm.decompose_affine(Aseq, vectors=True)
    sx_traj = np.cumprod(S[:,0])
    sy_traj = np.cumprod(S[:,1])
    r_traj  = np.cumsum(R)
    tx_traj = np.cumsum(T[:,0])
    ty_traj = np.cumsum(T[:,1])
    sx_smtraj = stabilize.smooth_motion(sx_traj, radius=args.radius)
    sy_smtraj = stabilize.smooth_motion(sy_traj, radius=args.radius)
    r_smtraj  = stabilize.smooth_motion(r_traj,  radius=args.radius)
    tx_smtraj = stabilize.smooth_motion(tx_traj, radius=args.radius)
    ty_smtraj = stabilize.smooth_motion(ty_traj, radius=args.radius)

    trajectory = sfm.compose_affine(sx_traj, sy_traj, r_traj, tx_traj, ty_traj)
    sm_trajectory = sfm.compose_affine(sx_smtraj, sy_smtraj, r_smtraj, tx_smtraj, ty_smtraj)
    print(Aseq.shape)
    vstab = transform.warp_sequence_affine(vz, Aseq+(sm_trajectory-trajectory))

    if args.visualize:
        visualize.plot_motion_affine(trajectory, sm_trajectory)
        visualize.interactive_play_video(vz, features=points, framerate=frate)
        visualize.interactive_play_video(vstab, framerate=frate)
    return vstab, sm_trajectory

if __name__ == '__main__':
    time_start = time.perf_counter()
    if args.out is None:
        if args.stabilize:
            outfile = 'stabilized.mp4'
        else:
            outfile = 'resampled.mp4'

    vs = cv2.VideoCapture(args.infile)
    nframes = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    hh      = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ww      = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    frate   = int(vs.get(cv2.CAP_PROP_FPS))

    logger.debug('Video Metadata:\n' + \
                 '  Frames:     {}\n'.format(nframes) + \
                 '  Size (WxH): {}x{}\n'.format(ww, hh) + \
                 '  Rate (fps): {}'.format(frate) )

    segment = None
    if args.fsegment or args.tsegment:
        if args.tsegment:
            segment = ( int(args.tsegment[0]//(1/frate)), int(args.tsegment[1]//(1/frate)) )
        else:
            segment = args.fsegment
        segment = (max(0, segment[0]), min(nframes, segment[1]))
        logger.debug('  Segment:    {}-{}'.format(*segment))
    logger.debug('')

    logger.info('Reading video sequence...')
    varr = video.read_all_frames(vs, segment=segment)

    logger.info('Resampling video sequence...')
    vz, ww, hh = video.resample_all_frames(varr, args.resample)

    if not args.stabilize:
        vstab = vz
    else:
        vg = video.grayscale_all_frames(varr)

        logger.info('Selecting tracking points...')
        points = sfm.calculate_features(vg) # shape: [nframes, npoints, dims=2]

        logger.info('Stabilizing video sequence...')
        if args.transform == 'perspective':
            vstab, sm_trajectory = stabilize_perspective(vz, vg, points)
        elif args.transform == 'affine':
            vstab, sm_trajectory = stabilize_affine(vz, vg, points)

    # save video out
    logger.info('Saving footage...')
    vout = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'mp4v'), frate, (ww, hh))
    for f in vstab:
        vout.write(f)
    logger.info('Processed footage saved to "{}"'.format(outfile))
    logger.debug('Total runtime: {:0.2f}s'.format(time.perf_counter()-time_start))
