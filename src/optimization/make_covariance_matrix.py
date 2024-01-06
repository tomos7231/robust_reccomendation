from __future__ import annotations

import numpy as np
import pandas as pd

from src.optimization.estimator import DiagonalEstimator, InputationEstimator
from src.paths import RESULT_DIR


def make_covariance_matrix(delta: float, exp_name: str, estimator: str) -> np.ndarray:
    if estimator == "DIAG":
        estimator = DiagonalEstimator(delta, exp_name)
    elif estimator == "INPUTE":
        estimator = InputationEstimator(delta, exp_name)
    else:
        raise Exception("Unknown estimator name: {}".format(estimator))

    cov_matrix = estimator.run()

    # 保存
    np.save(RESULT_DIR / exp_name / "cov_matrix.npy", cov_matrix)
