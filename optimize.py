import sys, os
import time
import math
from math import sqrt
import logging
import warnings

import numpy as np
from scipy import sparse
import numpy.linalg as linalg
import matplotlib.pyplot as plt

logger = logging.getLogger('vidstab.'+__name__)

sqeuclidean = lambda x: np.inner(x,x)

def nonneg(array):
    """projection of array onto non-negative orthant
    operation amounts to an elementwise max(0,val)
    """
    for x in np.nditer(array, op_flags=['readwrite'], order='K'):
        x[...] = max(0, x)
    return array

def proj_linf_ball(array, radius):
    """projection of array onto L_infinity norm ball (L_1 dual norm) of radius
    operation amounts to an elementwise min(radius,val)
    """
    for x in np.nditer(array, op_flags=['readwrite'], order='K'):
        x[...] = max(-radius, min(radius, x))
    return array

def soft_threshold(x, t):
    """soft thresholding operator (Prox operator for L1 norm)"""
    return np.sign(x)*np.maximum(np.abs(x)-t, 0)

def prox_0(x, t):
    return x

def prox_box_constraint(x, l, u):
    """Compute prox of indicator of box constraint, defined by lower and upper bound"""
    xc = np.copy(x)
    xc[xc<l] = l
    xc[xc>u] = u
    return xc


matD1 = None
def D1(x):
    """compute transpose of first order discrete (forward) difference of x"""
    # TODO: make more efficient by vector offset subtraction
    global matD1
    if matD1 is None:
        N = len(x)
        matD1 = -np.eye(N) + np.eye(N, k=1)
        matD1[-1:] = np.zeros((1, N))
    return matD1.dot(x)

matD1T = None
def D1T(x):
    """compute transpose of first order discrete (forward) difference of x"""
    # TODO: make more efficient by vector offset subtraction
    global matD1T
    if matD1T is None:
        N = len(x)
        matD1T = -np.eye(N) + np.eye(N, k=-1)
        matD1T[:, -1:] = np.zeros((N,1))
    return matD1T.dot(x)

matD2 = None
def D2(x):
    """compute transpose of second order discrete (forward) difference of x"""
    # TODO: make more efficient by vector offset subtraction
    global matD2
    if matD2 is None:
        N = len(x)
        matD2 = np.eye(N) + -2*np.eye(N, k=1) + np.eye(N, k=2)
        matD2[-2:] = np.zeros((1, N))
    return matD2.dot(x)

matD2T = None
def D2T(x):
    """compute transpose of second order discrete (forward) difference of x"""
    # TODO: make more efficient by vector offset subtraction
    global matD2T
    if matD2T is None:
        N = len(x)
        matD2T = np.eye(N) + -2*np.eye(N, k=-1) + np.eye(N, k=-2)
        matD2T[:, -2:] = np.zeros((N,1))
    return matD2T.dot(x)

def huber(x, mu):
    xc = np.copy(x)
    absxc = np.abs(xc)
    mask = (np.abs(xc)>mu)
    xc[mask] = absxc[mask] - 0.5*mu
    xc[~mask] = np.power(xc[~mask], 2)/(2*mu)
    return np.sum(np.abs(xc))
def grad_huber(x, mu):
    """compute clip function (gradient of huber loss), (projection onto sym. range set)"""
    xc = np.copy(x)
    mask = (np.abs(xc)>mu)
    xc[mask] = mu*np.sign(xc[mask])
    return xc

def forw_L2_huber(x, xhat, lamb, mu):
    """Compute forward operation of 1/2*||x-xhat||^2 + lambda*||Dx||_1
    Args:
        x:      opt. var
        xhat:   target (constant)
        lamb:   weighting for Huber regularization term
        mu:     huber smoothing coeff.
    """
    return 0.5*sqeuclidean(x-xhat) + lamb*huber(D2(x), mu)
def grad_L2_huber(x, xhat, lamb, mu):
    """Compute gradient of 1/2*||x-xhat||^2 + lambda*||Dx||_1
    Args:
        x:      opt. var
        xhat:   target (constant)
        lamb:   weighting for Huber regularization term
        mu:     huber smoothing coeff.
    """
    grad = (x-xhat) + (lamb/mu)*D2T(grad_huber(D2(x), mu))
    return grad

def forw_L2_huber_2(x, xhat, lamb1, lamb2, mu):
    """Compute forward operation of 1/2*||x-xhat||^2 + lambda*||Dx||_1
    Args:
        x:      opt. var
        xhat:   target (constant)
        lamb:   weighting for Huber regularization term
        mu:     huber smoothing coeff.
    """
    return 0.5*sqeuclidean(x-xhat) + lamb1*huber(D1(x), mu) + lamb2*huber(D2(x), mu)
def grad_L2_huber_2(x, xhat, lamb1, lamb2, mu):
    """Compute gradient of 1/2*||x-xhat||^2 + lambda*||Dx||_1
    Args:
        x:      opt. var
        xhat:   target (constant)
        lamb:   weighting for Huber regularization term
        mu:     huber smoothing coeff.
    """
    grad = (x-xhat) + (lamb1/mu)*D1T(grad_huber(D1(x), mu)) + (lamb2/mu)*D2T(grad_huber(D2(x), mu))
    return grad

def forw_L2_huber_3(x, xhat, lamb, mu):
    """Compute forward operation of 1/2*||x-xhat||^2 + lambda*||Dx||_1
    Args:
        x:      opt. var
        xhat:   target (constant)
        lamb:   weighting for Huber regularization term
        mu:     huber smoothing coeff.
    """
    return 0.5*sqeuclidean(x-xhat) + lamb*huber(D1(x), mu)
def grad_L2_huber_3(x, xhat, lamb, mu):
    """Compute gradient of 1/2*||x-xhat||^2 + lambda*||Dx||_1
    Args:
        x:      opt. var
        xhat:   target (constant)
        lamb:   weighting for Huber regularization term
        mu:     huber smoothing coeff.
    """
    grad = (x-xhat) + (lamb/mu)*D1T(grad_huber(D1(x), mu))
    return grad

def positive_root(t, tprev, thetaprev):
    """compute positive root of: tprev*theta^2 = t*thetaprev^2 * (1-theta)"""
    lhs = -t*thetaprev**2
    rhs = sqrt((t**2) * (thetaprev**4) - 4*tprev*t*(thetaprev**2))
    roots = (lhs + np.array((1, -1))*rhs)/(2*tprev)
    return roots[roots>0]

def positive_root_2(t, tprev, thetaprev):
    r = tprev/t
    return thetaprev*(sqrt(4*r+thetaprev**2) - thetaprev)/(2*r)

def positive_root_3(t, tprev, thetaprev):
    return 0.5*(1 + sqrt(1 + 4*tprev**2))


def optimize(*args, **kwargs):
    return FISTA_method1(*args, **kwargs)

def FISTA_method1(xhat, forwg, gradg, proxh, eps=1e-15, niters=100):
    logger.debug('Preparing for optimization...')
    time_start = time.time()

    # hyperparams
    beta1 = 3   # >=1
    beta2 = 0.4 # <1

    # initialize opt vars
    nu = xprev = xhat
    tprev = 1

    res_history = []
    f_history = []
    for kk in range(1,niters+1):
        ll = 0
        t = tprev*beta1
        theta = 2/(kk+1)
        y = (1-theta)*xprev + theta*nu
        while True:
            x = proxh(y-t*gradg(y) , t)

            # Lipschitz/step-size Condition
            ubound_gap = (forwg(y) + np.inner(gradg(y), x-y) + (1/(2*t))*sqeuclidean(x-y)) - forwg(x)
            #  logger.debug('k_{}|ls_{}>> t:{}, gap:{}, '.format(kk, ll, t, ubound_gap))
            if ubound_gap >= 0:
                break
            ll+=1
            t *= beta2

        residual = np.linalg.norm(xprev-x)/np.linalg.norm(xprev)
        res_history.append(residual)

        if (kk%20)==0:
            logger.debug('k_{}>> res:{}'.format(kk, residual))
        f_history.append(forwg(x))
        #  if kk>1:
        #      assert(f_history[-1]<=f_history[-2])
        if (residual <= eps):
            break

        nu = xprev + (1/theta)*(x-xprev)
        xprev = x

    return x

def FISTA_method2(xhat, forwg, gradg, proxh, eps=1e-15, niters=100):
    warnings.warn("FISTA_method2 is unstable and should not be used")
    logger.info('Preparing for optimization...')
    time_start = time.time()

    # hyperparams
    beta1 = 1 # >1
    beta2 = 0.5 # <1

    # initialize opt vars
    nuprev = xprev = np.random.rand(*xhat.shape)
    tprev = 10
    thetaprev = 1

    res_history = []
    f_history = []
    for kk in range(1,niters+1):
        ll = 0
        t = tprev*beta1
        while True:
            # line search
            theta = positive_root_2(t, tprev, thetaprev) if kk>1 else 1
            y = (1-theta)*xprev + theta*nuprev
            x = proxh(y-t*gradg(y) , t)

            # Lipschitz/step-size Condition
            ubound_gap = (forwg(y) + np.inner(gradg(y), x-y) + (1/(2*t))*sqeuclidean(x-y)) - forwg(x)
            logger.debug('k_{}|ls_{}>> t:{}, gap:{}, '.format(kk, ll, t, ubound_gap))
            if ubound_gap >= 0:
                break
            ll+=1
            t *= beta2

        residual = np.linalg.norm(xprev-x)/np.linalg.norm(xprev)
        if (kk%20)==0:
            logger.info('k_{}>> res:{}'.format(kk, residual))
        res_history.append(residual)
        f_history.append(forwg(x))
        if (residual <= eps):
            break

        nu = xprev + (1/theta)*(x-xprev)
        xprev = x
        nuprev = nu
        tprev = t
        thetaprev = theta


    #  plt.plot(res_history)
    #  plt.figure()
    #  plt.plot(f_history)
    #  plt.show()

    logger.debug(x)
    return x


if __name__ == '__main__':
    pass
    #  # huber test
    #  x = np.arange(-4, 4, 0.1)
    #  y = np.empty_like(x)
    #  yg = np.empty_like(x)
    #  for ii, xx in enumerate(x):
    #      y[ii] = huber(xx, 0.3)
    #      yg[ii] = grad_huber(xx, 0.3)
    #  plt.plot(x, y)
    #  plt.plot(x, yg)
    #  plt.show()

    #  # Optimization test
    #  logger.setLevel(logging.DEBUG)
    #  logger.addHandler(logging.StreamHandler(sys.stdout))
    #  xhat = np.zeros((10000,))
    #  x = optimize(xhat, lambda x: 0.5*sqeuclidean(x-xhat), lambda x: x-xhat, prox_0,)
    #  logger.debug(x)
