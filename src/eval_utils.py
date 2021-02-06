'''
Authors: Alex Wong <alexw@cs.ucla.edu>, Mukund Mundhra <mukundmundhra@cs.ucla.edu>

If this code is useful to you, please consider citing the following paper:

A. Wong, M. Mundhra, and S. Soatto. Stereopagnosia: Fooling Stereo Networks with Adversarial Perturbations.
https://arxiv.org/pdf/2009.10142.pdf

@inproceedings{wong2021stereopagnosia,
  title={Stereopagnosia: Fooling Stereo Networks with Adversarial Perturbations},
  author={Wong, Alex and Mundhra, Mukund and Soatto, Stefano},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  year={2021}
}
'''
import numpy as np


def root_mean_sq_err(src, tgt):
    '''
    Root mean squared error

    Args:
        src : numpy
            source array
        tgt : numpy
            target array
    Returns:
        float : root mean squared error between source and target
    '''
    return np.sqrt(np.mean((src - tgt) ** 2))

def mean_abs_err(src, tgt):
    '''
    Mean absolute error

    Args:
        src : numpy
            source array
        tgt : numpy
            target array
    Returns:
        float : mean absolute error between source and target
    '''
    return np.mean(np.abs(src - tgt))

def mean_abs_rel_err(src, tgt):
    '''
    Mean absolute relative error (normalize absolulte error)

    Args:
        src : numpy
            source array
        tgt : numpy
            target array
    Returns:
        float : mean absolute relative error between source and target
    '''
    return np.mean(np.abs(src - tgt) / tgt)

def d1_error(src, tgt):
    '''
    D1 error reported for KITTI 2015. A pixel is considered to be correctly estimated if the disparity end-point error is < 3 px or < 5 %.
    Args:
        src : numpy
            source array
        tgt : numpy
            target array
    Returns:
        float : d1 error between source and target (percentage of pixels)
    '''

    E = np.abs(src - tgt)
    n_err = np.count_nonzero(np.logical_and((tgt > 0), np.logical_and(E > 3, (E/np.abs(tgt)) > 0.05)))
    n_total = np.count_nonzero(tgt > 0)
    return n_err/n_total

def lp_norm(T, p=1.0, axis=None):
    '''
    Computes the Lp-norm of a tensor

    Args:
        T : numpy
            tensor
        p : float
            norm to use
        axis : int
            axis/dim to compute norm
    Returns:
        float : Lp norm of tensor
    '''
    if p != 0 and axis is None:
        return np.mean(np.abs(T))
    else:
        if p != 0:
            return np.mean(np.sum(np.abs(T) ** p, axis=axis)**(1.0/p))
        else:
            return np.max(np.abs(T))
