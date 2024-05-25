from __future__ import annotations

from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from tqdm import tqdm


class ShrinkageEstimator(metaclass=ABCMeta):
    def __init__(self, delta: float):
        self.delta = delta

    def run(self) -> pd.DataFrame:
        # データの読み込み
        self.pred_rating_df = self.load_data()
        # 共分散行列の計算
        S, F = self.make_covariance_matrix()
        # 出現行列の計算
        freq_matrix = self.create_freq_matrix(self.pred_rating_df)
        # 共分散行列の推定
        cov_matrix = self.calc_weighted_sum(S, F, self.delta)
        return cov_matrix, freq_matrix

    @staticmethod
    def load_data() -> pd.DataFrame:
        pred_rating_df = pd.read_csv("./pred_rating.csv")
        return pred_rating_df

    @staticmethod
    def create_freq_matrix(df: pd.DataFrame) -> np.ndarray:
        # あるアイテムを評価したユーザー件数とあるアイテムの組み合わせを評価したユーザー件数が格納された行列を作成
        all_item_ids = range(df["item_id"].max() + 1)
        train_df = df[df["data_type"] == "train"].reset_index(drop=True)
        # ピボットテーブルを作成
        matrix_user_item = train_df.pivot_table(
            index="user_id", columns="item_id", aggfunc="size", fill_value=0
        )
        # item_idが連続していない場合は、全てのitem_idを含むようにする
        matrix_user_item = matrix_user_item.reindex(columns=all_item_ids, fill_value=0)
        # 行列の積を計算
        item_matrix = matrix_user_item.T.dot(matrix_user_item)

        return item_matrix.values

    @abstractmethod
    def make_covariance_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def calc_weighted_sum(S: np.ndarray, F: np.ndarray, delta: float) -> np.ndarray:
        # SとFを重み付けした共分散行列を計算
        return (1 - delta) * S + delta * F


class DiagonalEstimator(ShrinkageEstimator):
    def __init__(self, delta: float):
        super().__init__(delta)

    def make_covariance_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        # 全てのアイテムのid
        all_item_ids = range(self.pred_rating_df["item_id"].max() + 1)

        # data_type=trainだけを抽出
        train_df = self.pred_rating_df[self.pred_rating_df["data_type"] == "train"].reset_index(
            drop=True
        )

        # user*itemの行列を作成
        matrix_user_item = train_df.pivot_table(index="user_id", columns="item_id", values="rating")
        # trainだけのデータだとitemidが不連続になるので、全てのitemidを含むようにする
        matrix_user_item = matrix_user_item.reindex(columns=all_item_ids, fill_value=np.nan)

        # 共分散行列の計算
        S = matrix_user_item.cov().values
        # 欠損値を0にする
        S = np.nan_to_num(S, nan=0.0)
        # Sの対角成分のみを取り出す
        F = np.diag(np.diag(S))

        return S, F


class InputationEstimator(ShrinkageEstimator):
    def __init__(self, delta: float):
        super().__init__(delta)

    def make_covariance_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        # 全てのアイテムのid
        all_item_ids = range(self.pred_rating_df["item_id"].max() + 1)

        # data_type=trainだけを抽出
        train_df = self.pred_rating_df.query("data_type == 'train'").reset_index(drop=True)

        # user*itemの行列を作成
        matrix_user_item = train_df.pivot_table(index="user_id", columns="item_id", values="rating")
        # trainだけのデータだとitemidが不連続になるので、全てのitemidを含むようにする
        matrix_user_item = matrix_user_item.reindex(columns=all_item_ids, fill_value=np.nan)

        # 共分散行列の計算
        S = matrix_user_item.cov().values
        # 欠損値を0にする
        S = np.nan_to_num(S, nan=0.0)
        # 全てのデータで共分散行列を計算
        matrix_user_item_all = self.pred_rating_df.pivot_table(
            index="user_id", columns="item_id", values="rating"
        )
        # なくてもいいかも
        matrix_user_item_all = matrix_user_item_all.reindex(columns=all_item_ids, fill_value=np.nan)
        F = matrix_user_item_all.cov().values
        # 欠損値を0にする
        F = np.nan_to_num(F, nan=0.0)

        return S, F
