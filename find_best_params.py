import collections
import pickle
import time
from os import path
from copy import deepcopy
from typing import Tuple, Any, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import MGSurvE as srv

from generate_landscape import PATH

(ID, OUT_PTH) = ("QSTART", "./QSTART/")
GENERATIONS = 150

LandscapeType = srv.Landscape
LogbookType = pd.DataFrame


def _plot_line(
    fig: Any,
    logbook: LogbookType,
    line_type: str,
    color: str,
    add_label: bool = False
) -> None:
    plot_args = {
        "color": color
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

    # Create subplot for plotting lines of behavior
    (avg_fig, avg_ax) = plt.subplots(1, 1, figsize=(9, 6))
    avg_ax.set_xlabel("Generations")
    avg_ax.set_ylabel("Min value")

    pops = [12, 40]
    speeds = [(20, -5)]
    option = 1

    results: List[Dict[str, float]] = []
    colors = ["r", "g", "b", "y"]
    for pop in pops:
        for speed in speeds:
            min_acum = 0
            time_acum = 0
            for i in range(0, 5):
                start_pso = time.time()
                lnd_pso = deepcopy(landscape)
                (lnd, logbook) = srv.optimize_traps_pso(
                    lnd_pso,
                    pop_size=pop,
                    generations=GENERATIONS,
                    speed_min=speed[1],
                    speed_max=speed[0],
                    phi1=2,
                    phi2=2,
                    w_max=1.5,
                    w_min=0.9,
                )
                finish_pso = time.time()
                _plot_line(avg_ax, logbook, f"PSO-{option}", add_label=(i == 0), color=colors[option - 1])

                curr_min = min(logbook["min"])
                min_acum += curr_min
                time_acum += finish_pso - start_pso

            results.append({
                "option": option,
                "min": min_acum / 5,
                "exec_time": time_acum / 5,
                "pop": pop,
                "min_speed": speed[1],
                "max_speed": speed[0],
            })
            option += 1

    pd.DataFrame(results).to_csv("search_params.csv")

    avg_ax.legend()
    avg_fig.savefig(path.join(OUT_PTH, f"{ID}_average_search.png"), dpi=300)


if __name__ == "__main__":
    main()
