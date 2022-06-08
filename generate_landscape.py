import pickle

import pandas as pd
import MGSurvE as srv


ID = "landscape"
PATH = f"{ID}.pickle"


def main() -> None:
    pts_num = 200
    radii = (75, 100)
    xy = srv.ptsDonut(pts_num, radii).T
    points = pd.DataFrame({"x": xy[0], "y": xy[1], "t": [0]*pts_num})

    null_traps = [0, 0, 0, 0]
    traps = pd.DataFrame({
        "x": null_traps, "y": null_traps,
        "t": null_traps, "f": null_traps
    })
    t_ker = {
        0: {"kernel": srv.exponentialDecay, "params": {"A": .5, "b": .1}}
    }

    landscape = srv.Landscape(
        points,
        kernelParams={"params": srv.MEDIUM_MOV_EXP_PARAMS, "zeroInflation": .25},
        traps=traps, trapsKernels=t_ker
    )

    with open(PATH, "wb") as handle:
        pickle.dump(landscape, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
