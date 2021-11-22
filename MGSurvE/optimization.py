#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rand
from deap import base, creator, algorithms, tools


###############################################################################
# Fitness function
###############################################################################
def reshapeInCanonicalForm(tau, sitesN, trapsN):
    """ Reshapes a migration matrix into canonical form (deprecated).
    
    Parameters:
        tau (numpy array): Traps migration matrix.
        sitesN (int): Number of sites.
        trapsN (int): Number of traps.

    Returns:
        (numpy array): Reshaped matrix in canonical form.
    """
    canO = list(range(sitesN, sitesN+trapsN))+list(range(0, sitesN))
    tauCan = np.asarray([[tau[i][j] for j in canO] for i in canO])
    return tauCan


def getMarkovAbsorbing(tauCan, trapsN):
    """ Get Markov's absorbing states (deprecated).
    
    Parameters:
        tauCan (numpy array): Traps migration matrix in canonical form.
        trapsN (int): Number of traps.

    Returns:
        (numpy array): Time to fall into absorbing states from anywhere in landscape.
    """
    A = tauCan[trapsN:, :trapsN]
    B = tauCan[trapsN:, trapsN:]
    F = np.linalg.inv(np.subtract(np.identity(B.shape[0]), B))
    return F


def getFundamentalMatrix(tau, sitesN, trapsN):
    """ Get Markov's fundamental matrix.
    
    Equivalent to using reshapeInCanonicalForm and getMarkovAbsorbing (which
        should be deprecated).

    Parameters:
        tau (numpy array): Traps migration matrix in canonical form.
        sitesN (int): Number of sites.
        trapsN (int): Number of traps.

    Returns:
        (numpy array): Time to fall into absorbing states from anywhere in landscape.
    """
    Q = tau[:sitesN, :sitesN]
    R = tau[:sitesN, -trapsN:]
    I = np.identity(Q.shape[0])
    F = np.linalg.inv(np.subtract(I, Q))
    return F


def getFundamentalFitness(
        fundamentalMatrix, fitFuns={'outer': np.mean, 'inner': np.max}
    ):
    daysInSites = np.apply_along_axis(fitFuns['inner'], 1, fundamentalMatrix)
    daysTillTrapped = fitFuns['outer'](daysInSites)
    return daysTillTrapped


###############################################################################
# GA
###############################################################################
def initChromosome(trapsNum, coordsRange):
    """ Generates a random uniform chromosome for GA optimization.
    
    Parameters:
        trapsNum (int): Number of traps to lay down in the landscape.
        xRan (tuple of tuples of floats): XY Range for the coordinates.

    Returns:
        (list): List of xy coordinates for the traps' positions.
    """
    (xRan, yRan) = coordsRange
    xCoords = np.random.uniform(xRan[0], xRan[1], trapsNum)
    yCoords = np.random.uniform(yRan[0], yRan[1], trapsNum)
    chromosome = [val for pair in zip(xCoords, yCoords) for val in pair]
    return np.asarray(chromosome)

def genFixedTrapsMask(trapsFixed, dims=2):
    """ Creates a mask for the fixed traps (non-movable).
    
    Parameters:
        trapsFixed (bool numpy array): Boolean array with the traps that are not movable (lnd.trapsFixed).
        dims (int): Unused for now, but it's the number of dimensions for the landscape.

    Returns:
        (numpy array): Mask of the elements that can be moved in the GA operations.
    """
    dups = [list([not(i)])*dims for i in trapsFixed]
    dupsVct = [item for sublist in dups for item in sublist]
    return np.asarray(dupsVct)


def mutateChromosome(
        chromosome, fxdTrpsMsk,
        randFun=rand.normal, 
        randArgs={'loc': 0, 'scale': 0.1}
    ):
    """ Mutates a chromosome with a probability distribution based on the mutation mask.
    
    Parameters:
        chromosome (floats numpy array): GA's float chromosome generated by initChromosome.
        fxdTrpsMsk (bool numpy array): Array of bools that define which alleles can be mutated (1).
        randFun (function): Probability function for the mutation operation.
        randArgs (dict): Arguments to control the shape of the probability function.

    Returns:
        (numpy array): Selectively-mutated chromosome.
    """
    randDraw = randFun(size=len(chromosome), **randArgs)
    randMsk = randDraw * fxdTrpsMsk
    mutChrom = chromosome + randMsk
    return mutChrom
