#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
import pandas as pd
import numpy.random as rand
from os import path


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
        fundamentalMatrix, 
        fitFuns={'outer': np.mean, 'inner': np.max}
    ):
    """ Get fitness from Markov's fundamental matrix.

    Parameters:
        fundamentalMatrix (numpy array): Markov's fundamental matrix (calcFundamentalMatrix)
        fitFuns (dict): Dictionary containing the inner (row) and outer (col) operations for the fundamental matrix.

    Returns:
        (float): Summarized fitness function for the fundamental matrix.
    """
    daysInSites = np.apply_along_axis(fitFuns['inner'], 1, fundamentalMatrix)
    daysTillTrapped = fitFuns['outer'](daysInSites)
    return daysTillTrapped


###############################################################################
# GA
###############################################################################
def initChromosome(trapsCoords, fixedTrapsMask, coordsRange):
    """ Generates a random uniform chromosome for GA optimization.
    
    Parameters:
        trapsNum (int): Number of traps to lay down in the landscape.
        xRan (tuple of tuples of floats): XY Range for the coordinates.

    Returns:
        (list): List of xy coordinates for the traps' positions.
    """
    (xRan, yRan) = coordsRange
    trapsNum = trapsCoords.shape[0]
    chromosome = trapsCoords.flatten()
    for i in range(trapsNum):
        if fixedTrapsMask[i]:
            if (i % 2) != 0: 
                chromosome[i] = np.random.uniform(xRan[0], xRan[1], 1)[0]
            else:
                chromosome[i] = np.random.uniform(yRan[0], yRan[1], 1)[0]
    # xCoords = np.random.uniform(xRan[0], xRan[1], trapsNum)
    # yCoords = np.random.uniform(yRan[0], yRan[1], trapsNum)
    # chromosome = [val for pair in zip(xCoords, yCoords) for val in pair]
    return chromosome


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
        chromosome, fixedTrapsMask,
        randFun=rand.normal, 
        randArgs={'loc': 0, 'scale': 0.1}
    ):
    """ Mutates a chromosome with a probability distribution based on the mutation mask (in place).
    
    Parameters:
        chromosome (floats numpy array): GA's float chromosome generated by initChromosome.
        fxdTrpsMsk (bool numpy array): Array of bools that define which alleles can be mutated (1).
        randFun (function): Probability function for the mutation operation.
        randArgs (dict): Arguments to control the shape of the probability function.

    Returns:
        (numpy array list): Selectively-mutated chromosome.
    """
    randDraw = randFun(size=len(chromosome), **randArgs)
    randMsk = randDraw * fixedTrapsMask
    chromosome[:] = chromosome + randMsk
    return (chromosome, )


def cxBlend(
        ind1, ind2, 
        fixedTrapsMask, 
        alpha=.5
    ):
    """ Mates two chromosomes by "blend" based on the provided mask (in place).
    
    This implementation is similar to DEAP's cxBlend (https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.cxBlend). 
    Follow this link for the original code: https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py

    Parameters:
        ind1 (floats numpy array): GA's float chromosome generated by initChromosome.
        ind2 (floats numpy array): GA's float chromosome generated by initChromosome.
        fxdTrpsMsk (bool numpy array): Array of bools that define which alleles can be mutated (1).
        alpha (float): weight for each of the chromosomes.

    Returns:
        (list of chromosomes): Mated individuals.
    """
    (offA, offB) = (ind1[:], ind2[:])
    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        if fixedTrapsMask[i]:
            gamma = (1. + 2. * alpha) * random.random() - alpha
            offA[i] = (1. - gamma) * x1 + gamma * x2
            offB[i] = gamma * x1 + (1. - gamma) * x2
    (ind1[:], ind2[:]) = (offA[:], offB[:])
    return (ind1, ind2)


###############################################################################
# Fitness Functions
###############################################################################
def getDaysTillTrapped(
        landscape,
        fitFuns={'outer': np.mean, 'inner': np.max}
    ):
    """Gets the number of timesteps until a walker falls into a trap.

    Parameters:
        landscape (object): Landscape object to use for the analysis.
        fitFuns (dict): Dictionary with the outer (row) and inner (col) functions to use on the matrix.

    Returns:
        (float): Number of days for mosquitoes to fall into traps given the fitFuns.
    """
    funMat = getFundamentalMatrix(
        landscape.trapsMigration, landscape.pointNumber, landscape.trapsNumber
    )
    daysTillTrapped = getFundamentalFitness(funMat, fitFuns=fitFuns)
    return daysTillTrapped


def calcFitness(
        chromosome, 
        landscape=None,
        optimFunction=getDaysTillTrapped,
        optimFunctionArgs={'outer': np.mean, 'inner': np.max},
        dims=2
    ):
    """Calculates the fitness function of the landscape given a chromosome (in place, so not thread-safe).

    Parameters:
        chromosome (floats numpy array): GA's float chromosome generated by initChromosome.
        landscape (object): Landscape object to use for the analysis.
        optimFunction (function): Function that turns a matrix into a fitness value.
        optimFunctionArgs (dict): Dictionary with the outer (row) and inner (col) functions to use on the matrix.

    Returns:
        (tuple of floats): Landscape's fitness function.
    """
    candidateTraps = np.reshape(chromosome, (-1, dims))
    landscape.updateTrapsCoords(candidateTraps)
    fit = optimFunction(landscape, fitFuns=optimFunctionArgs)
    return (float(abs(fit)), )


###############################################################################
# Logging results
###############################################################################
def exportLog(
        logbook,
        outPath,
        filename
    ):
    """Dumps a dataframe with the report of the GA's history.

    Parameters:
        logbook (object): DEAP GA object.
        outPath (path): Path where the file will be exported.
        F_NAME (string): Filenamme (without extension).

    """
    log = pd.DataFrame(logbook)
    log.to_csv(path.join(outPath, filename)+'.csv', index=False)


def importLog(
        outPath,
        filename
    ):
    """Gets the number of timesteps until a walker falls into a trap.

    Parameters:
        LOG_PTH (path): Path where the file is stored.
        F_NAME (dict): Filename with extension.

    Returns:
        (pandas dataframe): GA optimization log.
    """
    df = pd.read_csv(path.join(LOG_PTH, filename+'.csv'))
    return df