from __future__ import annotations

import pickle
from typing import TypeVar

import pandas as pd

from src.evaluation.metric import diversity, precision

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
    precision = precision(items_recommended, pred_rating_df)
    logger.info(f"Precision: {precision:.3f}")

    # Diversityの計算
    diversity = diversity(items_recommended)
    logger.info(f"Diversity: {diversity:.3f}")
