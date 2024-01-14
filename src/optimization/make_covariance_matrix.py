from __future__ import annotations

import numpy as np

from src.optimization.estimator import DiagonalEstimator, InputationEstimator


def make_covariance_matrix(delta: float, estimator: str) -> np.ndarray:
    if estimator == "DIAG":
        estimator = DiagonalEstimator(delta)
    elif estimator == "INPUTE":
        estimator = InputationEstimator(delta)
    else:
        raise Exception("Unknown estimator name: {}".format(estimator))

    cov_matrix = estimator.run()

    # 保存
    np.save("./cov_matrix.npy", cov_matrix)
