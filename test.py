import pickle
import time
from os import path
from copy import deepcopy
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

import MGSurvE as srv

from generate_landscape import PATH

(ID, OUT_PTH) = ("QSTART", "./QSTART/")


def _use_pso(landscape: srv.Landscape) -> Tuple[pd.DataFrame, srv.Landscape, float]:
    start_pso = time.time()
    lnd_pso = deepcopy(landscape)
    (lnd, logbook) = srv.optimize_traps_pso(
        lnd_pso,
        pop_size=40,
        generations=150,
        speed_min=-20,
        speed_max=20,
        phi1=2,
        phi2=2,
        w_max=1.5,
        w_min=0.9,
    )
    finish_pso = time.time()
    # srv.exportLog(logbook, OUT_PTH, f"{ID}_PSO_LOG")
    return logbook, lnd, finish_pso - start_pso


def _use_ga(landscape: srv.Landscape) -> Tuple[pd.DataFrame, srv.Landscape, float]:
    start_ga = time.time()
    lnd_ga = deepcopy(landscape)
    (lnd, logbook) = srv.optimizeTrapsGA(
        lnd_ga,
        generations=500,
        pop_size="auto",
        mating_params="auto",
        mutation_params="auto",
        selection_params="auto",
    )
    finish_ga = time.time()
    # srv.exportLog(logbook, OUT_PTH, f"{ID}_LOG")
    return logbook, lnd, finish_ga - start_ga


def main() -> None:

    with open(PATH, "rb") as handle:
        landscape: srv.Landscape = pickle.load(handle)

    """
    (fig, ax) = plt.subplots(1, 1, figsize=(15, 15))
    landscape.plotSites(fig, ax, size=100)
    landscape.plotMaskedMigrationNetwork(fig, ax, alphaMin=.6, lineWidth=25)
    srv.plotClean(fig, ax, frame=False)
    fig.savefig(
        path.join(OUT_PTH, f"{ID}_ENV.png"),
        facecolor="w", bbox_inches="tight", pad_inches=0.1, dpi=300
    )
    plt.close("all")
    """

    start_pso = time.time()
    lnd_pso = deepcopy(landscape)
    (lnd, logbook) = srv.optimize_traps_pso(
        lnd_pso,
        pop_size=40,
        generations=150,
        speed_min=-20,
        speed_max=20,
        phi1=2,
        phi2=2,
        w_max=1.5,
        w_min=0.9,
    )
    finish_pso = time.time()
    srv.exportLog(logbook, OUT_PTH, f"{ID}_PSO_LOG")

    (fig, ax) = plt.subplots(1, 1, figsize=(15, 15))
    lnd.plotSites(fig, ax, size=100)
    lnd.plotMaskedMigrationNetwork(fig, ax, alphaMin=.6, lineWidth=25)
    lnd.plotTraps(fig, ax)
    srv.plotClean(fig, ax, frame=False)
    srv.plotFitness(fig, ax, min(logbook["min"]))
    fig.savefig(
        path.join(OUT_PTH, f"{ID}_PSO_TRP.png"),
        facecolor="w", bbox_inches="tight", pad_inches=0.1, dpi=300
    )
    plt.close("all")

    (fig, ax) = plt.subplots(1, 1, figsize=(15, 15))
    srv.plotMatrix(fig, ax, lnd.trapsMigration, vmax=1e-2, trapsNumber=landscape.trapsNumber)
    srv.plotClean(fig, ax, frame=False)
    fig.savefig(
        path.join(OUT_PTH, f"{ID}_PSO_MTX.png"),
        facecolor="w", bbox_inches="tight", pad_inches=0.1, dpi=300
    )

    ##########################

    start_ga = time.time()
    lnd_ga = deepcopy(landscape)
    (lnd, logbook) = srv.optimizeTrapsGA(
        lnd_ga, generations=500,
        pop_size="auto", mating_params="auto",
        mutation_params="auto", selection_params="auto"
    )
    finish_ga = time.time()
    srv.exportLog(logbook, OUT_PTH, f"{ID}_LOG")

    (fig, ax) = plt.subplots(1, 1, figsize=(15, 15))
    lnd.plotSites(fig, ax, size=100)
    lnd.plotMaskedMigrationNetwork(fig, ax, alphaMin=.6, lineWidth=25)
    lnd.plotTraps(fig, ax)
    srv.plotClean(fig, ax, frame=False)
    srv.plotFitness(fig, ax, min(logbook["min"]))
    fig.savefig(
        path.join(OUT_PTH, f"{ID}_TRP.png"),
        facecolor="w", bbox_inches="tight", pad_inches=0.1, dpi=300
    )
    plt.close("all")

    (fig, ax) = plt.subplots(1, 1, figsize=(15, 15))
    srv.plotMatrix(fig, ax, lnd.trapsMigration, vmax=1e-2, trapsNumber=landscape.trapsNumber)
    srv.plotClean(fig, ax, frame=False)
    fig.savefig(
        path.join(OUT_PTH, f"{ID}_MTX.png"),
        facecolor="w", bbox_inches="tight", pad_inches=0.1, dpi=300
    )

    print(f"PSO: {finish_pso - start_pso} seconds.")
    print(f"GA: {finish_ga - start_ga} seconds.")


if __name__ == "__main__":
    main()
