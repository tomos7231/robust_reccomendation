from __future__ import annotations

from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd

from src.paths import RESULT_DIR


class ShrinkageEstimator(metaclass=ABCMeta):
    def __init__(self, delta: float, exp_name: str):
        self.delta = delta
        self.exp_name = exp_name

    def run(self) -> pd.DataFrame:
        # データの読み込み
        self.pred_rating_df = self.load_data(self.exp_name)
        # 共分散行列の計算
        S, F = self.make_covariance_matrix()
        # 共分散行列の推定
        cov_matrix = self.estimate_covariance_matrix(S, F, self.delta)
        return cov_matrix

    @staticmethod
    def load_data(exp_name: str) -> pd.DataFrame:
        pred_rating_df = pd.read_csv(RESULT_DIR / exp_name / "pred_rating.csv")
        pred_rating_df["user_id"] = pred_rating_df["user_id"].astype(int)
        pred_rating_df["item_id"] = pred_rating_df["item_id"].astype(int)
        return pred_rating_df

    @abstractmethod
    def make_covariance_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def estimate_covariance_matrix(S: np.ndarray, F: np.ndarray, delta: float) -> np.ndarray:
        # SとFを重み付けした共分散行列を計算
        return (1 - delta) * S + delta * F


class DiagonalEstimator(ShrinkageEstimator):
    def __init__(self, delta: float, exp_name: str):
        super().__init__(delta, exp_name)

    def make_covariance_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        # 全てのアイテムのid
        all_item_ids = range(self.pred_rating_df["item_id"].max() + 1)

        # data_type=trainだけを抽出
        train_df = self.pred_rating_df.query("data_type == 'train'").reset_index(drop=True)

        # user*itemの行列を作成
        matrix_user_item = train_df.pivot_table(
            index="user_id", columns="item_id", values="rating", fill_value=0
        )
        # trainだけのデータだとitemidが不連続になるので、全てのitemidを含むようにする
        matrix_user_item = matrix_user_item.reindex(columns=all_item_ids, fill_value=0)

        # 共分散行列の計算
        S = matrix_user_item.cov().values
        # Sの対角成分のみを取り出す
        F = np.diag(np.diag(S))

        return S, F


class InputationEstimator(ShrinkageEstimator):
    def __init__(self, delta: float, exp_name: str):
        super().__init__(delta, exp_name)

    def make_covariance_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        # 全てのアイテムのid
        all_item_ids = range(self.pred_rating_df["item_id"].max() + 1)

        # data_type=trainだけを抽出
        train_df = self.pred_rating_df.query("data_type == 'train'").reset_index(drop=True)

        # user*itemの行列を作成
        matrix_user_item = train_df.pivot_table(
            index="user_id", columns="item_id", values="rating", fill_value=0
        )
        # trainだけのデータだとitemidが不連続になるので、全てのitemidを含むようにする
        matrix_user_item = matrix_user_item.reindex(columns=all_item_ids, fill_value=0)

        # 共分散行列の計算
        S = matrix_user_item.cov().values

        # 全てのデータで共分散行列を計算
        matrix_user_item_all = self.pred_rating_df.pivot_table(
            index="user_id", columns="item_id", values="rating", fill_value=0
        )
        # なくてもいいかも
        matrix_user_item_all = matrix_user_item_all.reindex(columns=all_item_ids, fill_value=0)
        F = matrix_user_item_all.cov().values

        return S, F
