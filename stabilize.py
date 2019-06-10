import logging

import numpy as np

import optimize

logger = logging.getLogger('vidstab.'+__name__)

def smooth_motion(motion, radius=10):
    """implements moving average"""
    shape = motion.shape[1:]
    flatmotion = motion.reshape((motion.shape[0], -1))

    window_size = 2*radius+1
    f = np.ones(window_size)/window_size
    flatmpad = np.lib.pad(flatmotion, ((radius, radius), (0,0)), 'edge')
    np.lib.pad

    flatmsmooth = np.empty_like(flatmpad)
    for pp in range(flatmotion.shape[1]):
        flatmsmooth[:,pp] = np.convolve(flatmpad[:,pp], f, mode='same')

    # unpad and reshape
    smoothmotion = (flatmsmooth[radius:-radius]).reshape(motion.shape[0], *shape)
    return smoothmotion

def optimize_motion(motion, smoothness=1, eps=1e-4, niters=300, lamb=0.5, mu=0.1):
    print(smoothness)
    sm_motion = optimize.optimize(motion, lambda x: optimize.forw_L2_huber(x, motion, smoothness, mu), lambda x: optimize.grad_L2_huber(x, motion, smoothness, mu), optimize.prox_0, eps=eps, niters=niters)
    return sm_motion

