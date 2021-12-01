#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import scipy.stats as stats
from scipy.optimize import fsolve
from sklearn.preprocessing import normalize
import MGSurvE.constants as cst

# https://en.wikipedia.org/wiki/Sigmoid_function
# https://en.wikipedia.org/wiki/Logistic_function
# https://dhemery.github.io/DHE-Modules/technical/sigmoid/


###############################################################################
# Migration Kernels
###############################################################################
def zeroInflatedLinearMigrationKernel(distMat, params=[.75, 1]):
    '''Calculates the zero-inflated linear distance-based movement probability.

    Args:
        distMat (numpy array): Distances matrix.

    Returns:
        numpy array: Migration matrix.
    '''
    coordsNum = len(distMat)
    migrMat = np.empty((coordsNum, coordsNum))
    for (i, row) in enumerate(distMat):
        for (j, dst) in enumerate(row):
            migrMat[i][j] = inverseLinearStep(dst, params=params)
        migrMat[i] = migrMat[i] / sum(migrMat[i])
    return migrMat


def truncatedExponential(distance, params=cst.AEDES_EXP_PARAMS):
    ''' Calculates the zero-inflated exponential distance-based movement probability.

    Args:
        distance (float): Distances matrix.
        params (list): Shape distribution parameters [rate, a, b].

    Returns:
        float: Migration probability.
    '''
    if(params[1] > params[2]):
        return None

    scale = 1.0/params[0]
    gA = stats.expon.cdf(params[1], scale=scale)
    gB = stats.expon.cdf(params[2], scale=scale)
    if np.isclose(gA, gB):
        return None

    densNum = stats.expon.pdf(distance, scale=scale)
    densDen = gB - gA

    return (densNum/densDen)


def zeroInflatedExponentialKernel(
        distMat, params=cst.AEDES_EXP_PARAMS, zeroInflation=.75
    ):
    '''Calculates the migration matrix using a zero-inflated exponential function.

    Args:
        distMat (numpy array): Distances matrix.
        params (list): Shape distribution parameters [rate, a, b].
        zeroInflation (float): Probability to stay in the same place.

    Returns:
        numpy array: Migration matrix.
    '''
    coordsNum = len(distMat)
    migrMat = np.empty((coordsNum, coordsNum))
    for (i, row) in enumerate(distMat):
        for (j, dst) in enumerate(row):
            if(i == j):
                migrMat[i][j] = 0
            else:
                migrMat[i][j] = truncatedExponential(dst, params=params)
        migrMat[i] = migrMat[i] / np.sum(migrMat[i]) * (1 - zeroInflation)
    np.fill_diagonal(migrMat, zeroInflation)
    tauN = normalize(migrMat, axis=1, norm='l1')
    return tauN


###############################################################################
# Exponential Decay
###############################################################################
def exponentialDecay(dist, A=1, b=1):
    '''Calculates the probability of moving between points.

    Args:
        dist (float): Distance between points.
        A (float): Maximum amplitude at distance 0.
        b (float): Decay rate (higher means tighter kernel).

    Returns:
        float: Movement probability.
    '''
    prob = A * math.exp(-b * dist)
    return prob


def sigmoidDecay(dist, A=1, rate=.5, x0=10):
    '''Calculates the probability of moving between points.

    Args:
        dist (float): Distance between points.
        A (float): Maximum amplitude at distance 0.
        rate (float): Logistic growth rate or steepness of the curve.
        x0 (float): The x value of the sigmoid's midpoint

    Returns:
        float: Movement probability.
    '''
    prob = A - A / (1 + math.e ** (-rate * (dist - x0)))
    return prob


###############################################################################
# Auxiliary
###############################################################################
def nSolveKernel(kernelDict, yVal, guess=0):
    '''Calculates the distance it takes for the kernel to match a given probability (yVar).

    Args:
        kernelDict (dict): Dictionary with the kernel info {'kernel', 'params'}.
        yVal (float): Probability for which we are solving the distance.
        guess (float): Initial guess for the distance.

    Returns:
        float: Distance for the probability value.
    '''
    (kFun, kPar) = (kernelDict['kernel'], kernelDict['params'])
    func = lambda delta : yVal-kFun(delta, **kPar)
    distance = fsolve(func, guess)
    return distance[0]