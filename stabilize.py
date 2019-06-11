import logging

import numpy as np

import optimize

logger = logging.getLogger('vidstab.'+__name__)

def movingavg(motion, *args, smoothing=10, **kwargs):
    """implements moving average"""
    radius=smoothing
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

def optimize_FISTA1(motion, *args, smoothness=1, eps=1e-4, niters=300, lamb=0.5, mu=0.1, boxbound=None, **kwargs):
    if boxbound is not None:
        proxh = lambda x, t: optimize.prox_box_constraint(x, *boxbound)
    else:
        proxh = optimize.prox_0

    sm_motion = optimize.optimize(motion, lambda x: optimize.forw_L2_huber(x, motion, smoothness, mu), lambda x: optimize.grad_L2_huber(x, motion, smoothness, mu), proxh, eps=eps, niters=niters)
    return sm_motion

def optimize_FISTA2(motion, *args, smoothness=1, eps=1e-4, niters=300, lamb=0.5, mu=0.1, boxbound=None, **kwargs):
    if boxbound is not None:
        proxh = lambda x, t: optimize.prox_box_constraint(x, *boxbound)
    else:
        proxh = optimize.prox_0

    sm_motion = optimize.optimize(motion, lambda x: optimize.forw_L2_huber_2(x, motion, smoothness, smoothness, mu), lambda x: optimize.grad_L2_huber_2(x, motion, smoothness, smoothness, mu), proxh, eps=eps, niters=niters)
    return sm_motion

def optimize_FISTA3(motion, *args, smoothness=1, eps=1e-4, niters=300, mu=0.1, boxbound=None, **kwargs):
    if boxbound is not None:
        proxh = lambda x, t: optimize.prox_box_constraint(x, *boxbound)
    else:
        proxh = optimize.prox_0

    sm_motion = optimize.optimize(motion, lambda x: optimize.forw_L2_huber_3(x, motion, smoothness, mu), lambda x: optimize.grad_L2_huber_3(x, motion, smoothness, mu), proxh, eps=eps, niters=niters)
    return sm_motion

smooth_motion = None
def init_smoothing(method):
    global smooth_motion
    if method == 'optimize_FISTA1':
        logger.info('Using Optimized Smoothing Model')
        smooth_motion = lambda motion, *args, **kwargs: optimize_FISTA1(motion, *args, **kwargs)
    elif method == 'optimize_FISTA2':
        logger.info('Using Optimized Smoothing Model')
        smooth_motion = lambda motion, *args, **kwargs: optimize_FISTA2(motion, *args, **kwargs)
    elif method == 'optimize_FISTA3':
        logger.info('Using Optimized Smoothing Model')
        smooth_motion = lambda motion, *args, **kwargs: optimize_FISTA3(motion, *args, **kwargs)
    elif method == 'movingavg':
        logger.info('Using Moving Average Smoothing Model')
        smooth_motion = lambda motion, *args, **kwargs: movingavg(motion, *args, **kwargs)
