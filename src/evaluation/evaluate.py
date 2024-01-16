from __future__ import annotations

import pickle
from typing import TypeVar

import pandas as pd

from src.evaluation.metric import calc_diversity, calc_precision

logger = TypeVar("logger")


def evaluate(logger: logger) -> None:
    """
    評価指標を計算する関数
    """
    # データの読み込み
    pred_rating_df = pd.read_csv("./pred_rating.csv")
    with open("./items_recommended.pkl", "rb") as f:
        items_recommended = pickle.load(f)

    # Precisionの計算
    precision = calc_precision(items_recommended, pred_rating_df)
    logger.info(f"Precision: {precision:.5f}")

    # Diversityの計算
    diversity = calc_diversity(items_recommended)
    logger.info(f"Diversity: {diversity}")
