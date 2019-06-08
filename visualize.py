import numpy as np
import cv2
import matplotlib.pyplot as plt

from sfm import decompose_affine

def plot_motion(motion):
    """motion is an Nx3x3 homograph matrix for N frames describing the motion between each
    successive frame as:
    [  ]"""
    for ii in range(motion.shape[1]):
        for jj in range(motion.shape[2]):
             plt.plot(motion[:,ii, jj], label='M[{},{}]'.format(ii+1, jj+1))
    plt.title('motion trajectories')
    plt.legend()
    plt.xlabel('frame #')
    plt.show()

def plot_motion_affine(motion):
    """motion is an Nx3x3 homograph matrix for N frames describing the motion between each
    successive frame as:
    [  ]"""
    (S, R, T) = decompose_affine(motion, vectors=True)
    plt.plot(S[:,0], label='Sx')
    plt.plot(S[:,1], label='Sy')
    plt.plot(R,      label='R')
    plt.plot(T[:,0], label='Tx')
    plt.plot(T[:,1], label='Ty')
    plt.title('motion trajectories')
    plt.legend()
    plt.xlabel('frame #')
    plt.show()

def interactive_play_video(varr, framerate=None, features=None):
    if not framerate:
        framerate = 24
    playing = True

    ii = 0
    while ii<varr.shape[0]:
        frame = varr[ii].copy()
        if features is not None:
            pts = features[ii]
            for pt in pts[np.random.choice(len(pts), min(20,len(pts)), replace=False)]:
                if len(pt)==2:
                    cv2.circle(frame, (pt[0], pt[1]), 5, (0,0,255), -1)
        cv2.imshow('features', frame)
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
                cv2.destroyWindow('features')
                return
            elif playing:
                break
        if ii == varr.shape[0]-1:
            ii = 0
            continue
        ii += 1
    cv2.destroyAllWindows()
