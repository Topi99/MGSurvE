#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
from os import path
from sys import argv
from copy import deepcopy
import matplotlib.pyplot as plt
from deap import base, creator, algorithms, tools
from compress_pickle import dump, load
import MGSurvE as srv
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

(ID, OUT_PTH) = ('GA_DEMO', './demos/')
srv.makeFolder(OUT_PTH)
###############################################################################
# Defining Landscape
###############################################################################
ptsNum = 100
radii = (75, 100)
xy = srv.ptsDonut(ptsNum, radii).T
points = pd.DataFrame({'x': xy[0], 'y': xy[1], 't': [0]*xy.shape[1]})
mKer = {'params': [.075, 1.0e-10, math.inf], 'zeroInflation': .75}
###############################################################################
# Defining Traps
###############################################################################
nullTraps = [0, 0, 0, 0]
traps = pd.DataFrame({
    'x': nullTraps, 'y': nullTraps,
    't': nullTraps, 'f': nullTraps
})
tKer = {0: {'kernel': srv.exponentialDecay, 'params': {'A': .5, 'b': .1}}}
###############################################################################
# Setting Landscape Up
###############################################################################
lnd = srv.Landscape(
    points, kernelParams=mKer,
    traps=traps, trapsKernels=tKer
)
bbox = lnd.getBoundingBox()
trpMsk = srv.genFixedTrapsMask(lnd.trapsFixed)

# srv.initChromosome(lnd.trapsCoords, trpMsk, bbox)

trapsCoords = lnd.trapsCoords
coordsRange = bbox
fixedTrapsMask = trpMsk

(xRan, yRan) = coordsRange
trapsNum = trapsCoords.shape[0]
chromosome = trapsCoords.flatten().astype(float)
allele = 0
for trap in range(trapsNum):
    if fixedTrapsMask[allele]:
        chromosome[allele+0] = np.random.uniform(xRan[0], xRan[1], 1)
        chromosome[allele+1] = np.random.uniform(yRan[0], yRan[1], 1)
    allele = allele + 2
    print(allele)



(fig, ax) = plt.subplots(1, 1, figsize=(15, 15), sharey=False)
lnd.plotSites(fig, ax, size=100)
lnd.plotMigrationNetwork(fig, ax, alphaMin=.6, lineWidth=25)
srv.plotClean(fig, ax, frame=False)
fig.savefig(
    path.join(OUT_PTH, '{}_LND.png'.format(ID)), 
    facecolor='w', bbox_inches='tight', pad_inches=0.1, dpi=300
)
###############################################################################
# GA Settings
############################################################################### 
POP_SIZE = int(10*(lnd.trapsNumber*1.25))
(GENS, MAT, MUT, SEL) = (
    200,
    {'mate': .3, 'cxpb': 0.5}, 
    {'mean': 0, 'sd': min([i[1]-i[0] for i in bbox])/5, 'mutpb': .5, 'ipb': .5},
    {'tSize': 3}
)
lndGA = deepcopy(lnd)
###############################################################################
# Registering Functions for GA
############################################################################### 
(lnd, logbook) = srv.optimizeTrapsGA(
        lndGA, pop_size='auto', generations=GENS,
        mating_params=MAT, mutation_params=MUT, selection_params=SEL,
        fitFuns={'outer': np.mean, 'inner': np.max}, verbose=True
    )
srv.exportLog(logbook, OUT_PTH, '{}_LOG'.format(ID))
###############################################################################
# Plot Landscape
############################################################################### 
(fig, ax) = plt.subplots(1, 1, figsize=(15, 15), sharey=False)
lnd.plotSites(fig, ax, size=100)
lnd.plotMigrationNetwork(fig, ax, alphaMin=.6, lineWidth=25)
lnd.plotTraps(fig, ax)
srv.plotClean(fig, ax, frame=False)
srv.plotFitness(fig, ax, min(logbook['min']))
fig.savefig(
    path.join(OUT_PTH, '{}_TRP.png'.format(ID)), 
    facecolor='w', bbox_inches='tight', pad_inches=0.1, dpi=300
)
plt.close('all')
###############################################################################
# Plot GA
############################################################################### 
(fig, ax) = plt.subplots(figsize=(15, 15))
(fig, ax) = srv.plotGAEvolution(fig, ax, logbook)
pthSave = path.join(
    OUT_PTH, '{}_GAP'.format(ID)
)
fig.savefig(
    pthSave,
    facecolor='w', bbox_inches='tight', 
    pad_inches=.1, dpi=300
)