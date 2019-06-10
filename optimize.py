import sys, os
import time
import math
from math import sqrt
import logging

import numpy as np
from scipy import sparse
import numpy.linalg as linalg
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

sqeuclidean = lambda x: np.inner(x,x)

def makeDiscreteDiffMatrix(out_shape, shape, dir='+x', makesparse=False):
    r, c = shape
    n = out_shape[0]

    # define how to assign values and how data is stored
    if not makesparse:
        D = np.zeros((n,n))
        def assign(r, c, val):
            D[r, c] = val
    else:
        # build coordinate and data lists
        data = []
        r_ind = []
        c_ind = []
        def assign(r, c, val):
            r_ind.append(r)
            c_ind.append(c)
            data.append(val)

    # construct matrix
    if dir == '+x':
        for y in range(n):
            if y<n-1 and (y==0 or not (y+1)%c==0):
                assign(y,y,1)
                assign(y,y+1,-1)
            else:
                assign(y,y,1)
                assign(y,y-1,-1)
    elif dir == '+y':
        for y in range(n):
            if y<n-c and (y<(c*(r-1))):
                assign(y,y,1)
                assign(y,y+c,-1)
            else:
                assign(y,y,1)
                assign(y,y-c,-1)

    if makesparse:
        D = sparse.coo_matrix((data, (r_ind, c_ind)), shape=(n,n))
    return D

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

def D2(x):
    """compute transpose of second order discrete (forward) difference of x"""
    # TODO: make more efficient by vector offset subtraction
    N = len(x)
    D = np.eye(N) + -2*np.eye(N, k=1) + np.eye(N, k=2)
    D[-2:] = np.zeros((1, N))
    return D.dot(x)

def D2T(x):
    """compute transpose of second order discrete (forward) difference of x"""
    # TODO: make more efficient by vector offset subtraction
    N = len(x)
    DT = np.eye(N) + -2*np.eye(N, k=-1) + np.eye(N, k=-2)
    DT[:, -2:] = np.zeros((N,1))
    return DT.dot(x)

def huber(x, mu):
    xc = np.copy(x)
    absxc = np.abs(xc)
    mask = (np.abs(xc)>mu)
    xc[mask] = absxc[mask] - 0.5*mu
    xc[~mask] = np.power(xc[~mask], 2)/(2*mu)
    return xc
def grad_huber(x, h):
    """compute clip function (gradient of huber loss), (projection onto sym. range set)"""
    xc = np.copy(x)
    mask = (np.abs(xc)>h)
    xc[mask] = h*np.sign(xc[mask])
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

def positive_root(t, tprev, thetaprev):
    """compute positive root of: tprev*theta^2 = t*thetaprev^2 * (1-theta)"""
    print(t, tprev, thetaprev)
    lhs = -t*thetaprev**2
    rhs = sqrt((t**2) * (thetaprev**4) - 4*tprev*t*(thetaprev**2))
    roots = (lhs + np.array((1, -1))*rhs)/(2*tprev)
    print(roots)
    return roots[roots>0]

def positive_root_2(t, tprev, thetaprev):
    r = tprev/t
    return thetaprev*(sqrt(4*r+thetaprev**2) - thetaprev)/(2*r)

def optimize(xhat, forwg, gradg, proxh, wreg=1.0, eps=1e-15, niters=100):
    logger.info('Preparing for optimization...')
    time_start = time.time()

    # initialize opt vars
    nu = x = np.random.rand(*xhat.shape)
    #  x = xhat
    #  nu = xhat
    beta1 = 2 # >1
    beta2 = 4 # >1
    theta = 1 #==1
    tprev = 0 #==0
    t = 1

    res_history = []
    f_history = []
    for kk in range(1,niters+1):
        # FISTA UPDATE STEP
        xprev = x
        nuprev = nu
        tprev = t if kk>1 else 0
        thetaprev = theta if kk>1 else 1

        # line search
        t *= beta1
        ll = 0
        while True:
            theta = positive_root_2(t, tprev, thetaprev) if kk>1 else 1
            y = (1-theta)*xprev + theta*nuprev
            x = proxh(y-t*gradg(y) , t)

            # check inequality
            ubound_gap = ((forwg(y) + np.inner(gradg(y), x-y) + 0.5*t*sqeuclidean(x-y))) - forwg(x)
            logger.debug('k_{}|ls_{}>> t:{}, gap:{}, '.format(kk, ll, t, ubound_gap))
            if ubound_gap >= 0:
                break
            ll+=1
            t /= beta2

        nu = xprev + (1/theta)*(x-xprev)
        residual = np.linalg.norm(xprev-x)/np.linalg.norm(xprev)
        logger.debug('k_{}>> res:{}'.format(kk, residual))
        res_history.append(residual)
        f_history.append(forwg(x))
        if (residual <= eps):
            break

    plt.plot(res_history)
    plt.figure()
    plt.plot(f_history)
    plt.show()

    return x


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    xhat = np.zeros((10000,))
    x = optimize(xhat, lambda x: 0.5*sqeuclidean(x-xhat), lambda x: x-xhat, prox_0,)
    logger.debug(x)




def optimizeFluence_CP(reduced_dose_matrix_list, ptv_low, roi_high_list, obj_wts, fluence_map_shape, TV_reg=1.0, costevery=25, niters=200, eps=3e-4, resume=False):
    logger.info('Preparing for optimization...')
    time_start = time.time()
    f_checkpt = 'p2_checkpt_opt_wts.pickle'

    # get parameter vector sizes and initialize variables
    init_factor = 1.0
    nx = reduced_dose_matrix_list[0].shape[1] # number of beamlets
    nROIs = len(reduced_dose_matrix_list)
    x = init_factor * np.ones((nx, 1))
    x_bar = init_factor * np.ones((nx, 1))

    nz1 = reduced_dose_matrix_list[0].shape[0] # A0 nvoxels
    z1 = init_factor * np.ones((nz1,1))
    z2i = []
    for mat in reduced_dose_matrix_list:
        nz2i = mat.shape[0] # Ai nvoxels for i=0..N (ptv and oars)
        z2i.append(init_factor * np.ones((nz2i,1)))
    z3 = np.ones((nx,1))
    z4 = np.ones((nx,1))

    # initialize hyper-parameters
    p = 1.1  # over-relaxation parameter
    gamma = TV_reg

    # construct discrete gradient matrices
    nbeams = int(nx / np.product(fluence_map_shape))
    d_len = nbeams * np.product(fluence_map_shape)
    Dx = makeDiscreteDiffMatrix((d_len,d_len), fluence_map_shape, dir='+x')
    Dy = makeDiscreteDiffMatrix((d_len,d_len), fluence_map_shape, dir='+y')

    # build K
    K = np.concatenate([reduced_dose_matrix_list[0]] +
                       [rdm for rdm in reduced_dose_matrix_list] +
                       [Dx, Dy],
                       axis=0)
    rowsum_k = np.sum(np.absolute(K), axis=1)
    sigmas = np.true_divide(1, rowsum_k+1e-9).reshape((-1,1))
    colsum_k = np.sum(np.absolute(K), axis=0)
    taus = np.true_divide(1, colsum_k+1e-9).reshape((-1,1))

    time_stop = time.time()
    logger.info('Setup complete. time elapsed: {!s}'.format(time.strftime('%M:%S', time.gmtime(time_stop-time_start))))

    # resume from last saved checkpoint?
    if resume and os.path.exists(f_checkpt):
        try:
            with open(f_checkpt, mode='rb') as f:
                x = pickle.load(f)
            logger.info('Resuming from file {!s}.'.format(f_checkpt))
        except:
            logger.info('Resume with file {!s} failed, starting from initialized variables.'.format(f_checkpt))

    # iterate
    logger.info('Beginning Optimization')
    logger.info('------------------------------------------')
    time_start = time.time()
    costs = []
    stop_flag = False
    for k in range(1,niters+1):
        status_update = False
        if (k==1 or k%costevery==0 or k==niters): status_update = True
        if (status_update or k%5==0):
            logger.info('iteration {:d} (checkpoint saved)'.format(k))
            with open(f_checkpt, mode='wb') as f:
                pickle.dump(x, f)

        # update variables
        #------------------------------
        z = np.concatenate([z1] +
                           [zi for zi in z2i] +
                           [z3, z4],
                           axis=0)
        x_bar_next = nonneg(x - np.multiply(taus, (K.transpose().dot(z))))
        #------------------------------
        zhat = z + sigmas*K.dot(2*x_bar_next - x)

        # One-sided L2 PTV dose prescription fidelity term
        z1_bar_next = np.empty_like(z1)
        for j in range(z1.shape[0]):
            numer = ptv_low - (zhat[j]/sigmas[j])
            if (numer > 0):
                beta_j = numer/(1+(obj_wts[0]/sigmas[j]))
            else:
                beta_j = numer
            z1_bar_next[j] = zhat[j] - sigmas[j]*(ptv_low - beta_j)
        #------------------------------
        # One-sided L2 OAR dose limiting term
        z2i_bar_next = []
        idx = z1.shape[0]
        for i in range(nROIs):
            this_z2i_bar_next = np.empty_like(z2i[i])
            for j in range(z2i[i].shape[0]):
                numer = (zhat[idx+j]/sigmas[idx+j]) - roi_high_list[i]
                if (numer > 0):
                    beta_j = numer/(1+(obj_wts[i+1]/sigmas[idx+j]))
                else:
                    beta_j = numer
                this_z2i_bar_next[j] = zhat[idx+j] - sigmas[idx+j]*(beta_j + roi_high_list[i])
            idx += z2i[i].shape[0]
            z2i_bar_next.append(this_z2i_bar_next)
        #------------------------------
        # TV Regularization
        z4_bar_next = proj_linf_ball( zhat[idx:idx+z3.shape[0]] , gamma)
        z3_bar_next = proj_linf_ball( zhat[idx+z3.shape[0]:] , gamma)
        #------------------------------
        #------------------------------
        # Over-relaxation updates
        x_next = p*x_bar_next + (1.0-p)*x
        z1_next = p*z1_bar_next + (1.0-p)*z1
        z2i_next = []
        for i in range(nROIs):
            z2i_next.append(p*z2i_bar_next[i] + (1.0-p)*z2i[i])
        z3_next = p*z3_bar_next + (1.0-p)*z3
        z4_next = p*z4_bar_next + (1.0-p)*z4
        #------------------------------

        # check residual for convergence criterion
        if (status_update):
            r1 = np.true_divide((x - x_next), taus+1e-9)
            z_next = np.concatenate([z1_next] +
                                   [zi for zi in z2i_next] +
                                   [z3_next, z4_next],
                                   axis=0)
            # K.dot(x) multiplication is expensive
            r2 = np.true_divide((z - z_next), sigmas+1e-9) + K.dot(x_bar - x_next)

            rr1 = linalg.norm(r1)/linalg.norm(x_next)
            rr2 = linalg.norm(r2)/linalg.norm(z_next)
            logger.info('  |--> rel. residual 1: {:12.6e}'.format(rr1))
            logger.info('  |--> rel. residual 2: {:12.6e}'.format(rr2))

            if rr1<eps and rr2<eps:
                logger.info('STOPPING CRITERION MET')
                stop_flag = True

        # submit variable updates
        x = x_next
        z1 = z1_next
        for i in range(len(reduced_dose_matrix_list)):
            z2i[i] = z2i_next[i]
        z3 = z3_next
        z4 = z4_next

        # compute cost at checkpoints
        if (status_update):
            cost = obj_wts[0]/2 * pow(linalg.norm(nonneg(np.subtract(ptv_low, reduced_dose_matrix_list[0].dot(x)))),2) # PTV prescription fidelity term
            for i in range(len(reduced_dose_matrix_list)):
                cost += obj_wts[i+1]/2 * pow(linalg.norm(nonneg(reduced_dose_matrix_list[i].dot(x) - roi_high_list[i])),2) # Dose limits fidelity term
            cost += gamma*linalg.norm(Dx.dot(x), ord=1) + gamma*linalg.norm(Dy.dot(x), ord=1)

            logger.info('  |--> Cost: {:12.6e} '.format(cost))
            costs.append(cost)

        # break early if convergence criterion was met
        if stop_flag:
            break

    time_stop = time.time()
    logger.info('------------------------------------------')
    logger.info('Optimization complete. time elapsed: {!s}'.format(time.strftime('%M:%S', time.gmtime(time_stop-time_start))))

    return x, costs
