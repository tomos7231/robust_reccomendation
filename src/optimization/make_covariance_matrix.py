from __future__ import annotations

import numpy as np

from src.optimization.estimator import DiagonalEstimator, InputationEstimator


def make_covariance_matrix(delta: float, estimator: str) -> None:
    if estimator == "DIAG":
        estimator = DiagonalEstimator(delta)
    elif estimator == "INPUTE":
        estimator = InputationEstimator(delta)
    else:
        raise Exception("Unknown estimator name: {}".format(estimator))

    cov_matrix, freq_matrix = estimator.run()

    # 保存
    np.save("./cov_matrix.npy", cov_matrix)
    np.save("./freq_matrix.npy", freq_matrix)
