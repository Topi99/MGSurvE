import collections
import pickle
import time
from os import path
from copy import deepcopy
from typing import Tuple, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import MGSurvE as srv

from generate_landscape import PATH

(ID, OUT_PTH) = ("QSTART", "./QSTART/")
AMOUNT_OF_REPETITIONS = 1
GENERATIONS = 150

LandscapeType = srv.Landscape
LogbookType = pd.DataFrame


def _use_pso(landscape: LandscapeType) -> Tuple[LogbookType, LandscapeType, float]:
    start_pso = time.time()
    lnd_pso = deepcopy(landscape)
    (lnd, logbook) = srv.optimize_traps_pso(
        lnd_pso,
        pop_size=40,
        generations=GENERATIONS,
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


def _use_ga(landscape: LandscapeType) -> Tuple[LogbookType, LandscapeType, float]:
    start_ga = time.time()
    lnd_ga = deepcopy(landscape)
    (lnd, logbook) = srv.optimizeTrapsGA(
        lnd_ga,
        generations=GENERATIONS - 1,
        pop_size="auto",
        mating_params="auto",
        mutation_params="auto",
        selection_params="auto",
    )
    finish_ga = time.time()
    # srv.exportLog(logbook, OUT_PTH, f"{ID}_LOG")
    return logbook, lnd, finish_ga - start_ga


def _plot_figures(landscape: LandscapeType, logbook: LogbookType, suffix: str) -> None:
    (fig, ax) = plt.subplots(1, 1, figsize=(15, 15))
    landscape.plotSites(fig, ax, size=100)
    landscape.plotMaskedMigrationNetwork(fig, ax, alphaMin=.6, lineWidth=25)
    landscape.plotTraps(fig, ax)
    srv.plotClean(fig, ax, frame=False)
    srv.plotFitness(fig, ax, min(logbook["min"]))
    fig.savefig(
        path.join(OUT_PTH, f"{ID}_TRP_{suffix}.png"),
        facecolor="w", bbox_inches="tight", pad_inches=0.1, dpi=300
    )
    plt.close("all")

    (fig, ax) = plt.subplots(1, 1, figsize=(15, 15))
    srv.plotMatrix(
        fig, ax, landscape.trapsMigration, vmax=1e-2, trapsNumber=landscape.trapsNumber
    )
    srv.plotClean(fig, ax, frame=False)
    fig.savefig(
        path.join(OUT_PTH, f"{ID}_MTX_{suffix}.png"),
        facecolor="w", bbox_inches="tight", pad_inches=0.1, dpi=300
    )


def _plot_line(
    fig: Any,
    logbook: LogbookType,
    line_type: str,
    add_label: bool = False
) -> None:
    color = "b" if line_type == "PSO" else "r"
    plot_args = {
        "color": color, "alpha": 0.14
    }
    if add_label:
        plot_args["label"] = line_type
    fig.plot(logbook["min"], **plot_args)


def _find_best_and_worst(
    curr_best: float,
    global_min: float,
    global_max: float,
    landscape: LandscapeType,
    logbook: LogbookType,
    curr_best_landscape: LandscapeType,
    curr_best_logbook: LogbookType,
    curr_worst_landscape: LandscapeType,
    curr_worst_logbook: LogbookType,
) -> Tuple[LandscapeType, LogbookType, LandscapeType, LogbookType, float, float]:
    """Updates variables with best and worst cases.

    Args:
        curr_best: The current minimum
        global_min: The global best
        global_max: The global worst
        landscape: The current landscape
        logbook: The current logbook
        curr_best_landscape: The best found landscape
        curr_best_logbook: The best found logbook
        curr_worst_landscape: The worst found landscape
        curr_worst_logbook: The worst found logbook

    Returns:
        A Tuple of the new best landscape, the new best logbook, the new worst landscape,
            the new worst logbook, the new best and the new worst.
    """
    new_best = global_min
    new_worst = global_max
    new_best_landscape = curr_best_landscape
    new_best_logbook = curr_best_logbook
    new_worst_landscape = curr_worst_landscape
    new_worst_logbook = curr_worst_logbook

    if curr_best < global_min:
        new_best_landscape = landscape
        new_best_logbook = logbook
        new_best = curr_best
    elif curr_best > global_max:
        new_worst_landscape = landscape
        new_worst_logbook = logbook
        new_worst = curr_best

    return (
        new_best_landscape,
        new_best_logbook,
        new_worst_landscape,
        new_worst_logbook,
        new_best,
        new_worst
    )


def main() -> None:
    with open(PATH, "rb") as handle:
        landscape: LandscapeType = pickle.load(handle)

    # (fig, ax) = plt.subplots(1, 1, figsize=(15, 15))
    # landscape.plotSites(fig, ax, size=100)
    # landscape.plotMaskedMigrationNetwork(fig, ax, alphaMin=.6, lineWidth=25)
    # srv.plotClean(fig, ax, frame=False)
    # fig.savefig(
    #     path.join(OUT_PTH, f"{ID}_ENV.png"),
    #     facecolor="w", bbox_inches="tight", pad_inches=0.1, dpi=300
    # )
    # plt.close("all")

    # Helper variables
    pso_time_acum = 0
    ga_time_acum = 0
    best_pso_landscape = None
    worst_pso_landscape = None
    best_ga_landscape = None
    worst_ga_landscape = None
    best_pso_logbook = None
    worst_pso_logbook = None
    best_ga_logbook = None
    worst_ga_logbook = None
    min_pso = float("inf")
    min_ga = float("inf")
    max_pso = 0
    max_ga = 0

    # Create subplot for plotting lines of behavior
    (avg_fig, avg_ax) = plt.subplots(1, 1, figsize=(9, 6))
    avg_ax.set_xlabel("Generations")
    avg_ax.set_ylabel("Min value")

    pso_aux_dict = collections.defaultdict(list)
    ga_aux_dict = collections.defaultdict(list)

    for i in range(0, AMOUNT_OF_REPETITIONS):
        print("="*5, f"Gen: {i}", "="*5)
        (pso_logbook, pso_landscape, pso_runtime) = _use_pso(landscape)
        _plot_line(avg_ax, pso_logbook, "PSO", add_label=(i == 0))
        pso_time_acum += pso_runtime
        for j, val in enumerate(pso_logbook["min"]):
            pso_aux_dict[j].append(val)

        curr_min_pso = min(pso_logbook["min"])

        (
            best_pso_landscape,
            best_pso_logbook,
            worst_pso_landscape,
            worst_pso_logbook,
            min_pso,
            max_pso,
        ) = _find_best_and_worst(
            curr_min_pso,
            min_pso,
            max_pso,
            pso_landscape,
            pso_logbook,
            best_pso_landscape,
            best_pso_logbook,
            worst_pso_landscape,
            worst_pso_logbook,
        )

        """
        (ga_logbook, ga_landscape, ga_runtime) = _use_ga(landscape)
        _plot_line(avg_ax, ga_logbook, "GA", add_label=(i == 0))
        ga_time_acum += ga_runtime
        for j, val in enumerate(ga_logbook["min"]):
            ga_aux_dict[j].append(val)
        curr_min_ga = min(ga_logbook["min"])

        (
            best_ga_landscape,
            best_ga_logbook,
            worst_ga_landscape,
            worst_ga_logbook,
            min_ga,
            max_ga,
        ) = _find_best_and_worst(
            curr_min_ga,
            min_ga,
            max_ga,
            ga_landscape,
            ga_logbook,
            best_ga_landscape,
            best_ga_logbook,
            worst_ga_landscape,
            worst_ga_logbook,
        )
        """

    """
    (fig_per, ax_per) = plt.subplots(1, 1, figsize=(9, 6))
    ax_per.set_xlabel("Generations")
    ax_per.set_ylabel("Min value")
    pso_ndarray = np.array([pso_aux_dict[key] for key in pso_aux_dict.keys()])
    pso_average = np.average(pso_ndarray, axis=1)
    pso_25th = np.percentile(pso_ndarray, 25, axis=1)
    pso_75th = np.percentile(pso_ndarray, 75, axis=1)

    ax_per.plot(pso_average, color="b", label="PSO")
    ax_per.fill_between(
        [i for i in range(0, GENERATIONS)],
        y1=pso_25th,
        y2=pso_75th,
        alpha=0.4, facecolor="b",
    )

    ga_ndarray = np.array([ga_aux_dict[key] for key in ga_aux_dict.keys()])
    ga_average = np.average(ga_ndarray, axis=1)
    ga_25th = np.percentile(ga_ndarray, 25, axis=1)
    ga_75th = np.percentile(ga_ndarray, 75, axis=1)

    ax_per.plot(ga_average, color="r", label="GA")
    ax_per.fill_between(
        [i for i in range(0, GENERATIONS)],
        y1=ga_25th,
        y2=ga_75th,
        alpha=0.4, facecolor="r",
    )
    ax_per.legend()
    fig_per.savefig(path.join(OUT_PTH, f"{ID}_percentile.png"), dpi=300)
    """
    # _plot_figures(best_ga_landscape, best_ga_logbook, "GA_BEST")
    _plot_figures(best_pso_landscape, best_pso_logbook, "PSO_BEST")

    """
    if worst_ga_landscape is None or worst_ga_logbook is None:
        print("Could not plot worst GA results")
    else:
        _plot_figures(worst_ga_landscape, worst_ga_logbook, "GA_WORST")

    if worst_pso_landscape is None or worst_pso_logbook is None:
        print("Could not plot worst PSO results")
    else:
        _plot_figures(worst_pso_landscape, worst_pso_logbook, "PSO_WORST")

    avg_ax.legend()
    avg_fig.savefig(path.join(OUT_PTH, f"{ID}_average.png"), dpi=300)
    """
    print(f"PSO: {pso_time_acum / AMOUNT_OF_REPETITIONS} seconds.")
    print(f"GA: {ga_time_acum / AMOUNT_OF_REPETITIONS} seconds.")


if __name__ == "__main__":
    main()
