from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.paths import DATA_DIR


class DataProcessor:
    def __init__(self, dataset_name: str, test_size: float, min_count_rating: int, seed: int = 42):
        self.dataset_name = dataset_name
        self.seed = seed
        self.test_size = test_size
        self.min_count_rating = min_count_rating
        self.seed = seed

    def run(self) -> None:
        # データの読み込み
        df = self.load_data(self.dataset_name)
        # データの前処理
        df = self.filter_data(df, self.min_count_rating)
        print(df.shape)
        # データの分割
        train_df, test_df = self.split_data(df, self.test_size, self.seed)
        return train_df, test_df

    @staticmethod
    def load_data(dataset_name: str) -> pd.DataFrame:
        if dataset_name == "movielens":
            df = pd.read_csv(
                DATA_DIR / "u.data", names=["user_id", "item_id", "rating", "timestamp"], sep="\t"
            )
            # timestamp is not used
            df = df.drop("timestamp", axis=1)
            # user_id and item_id start from 0
            df["user_id"] -= 1
            df["item_id"] -= 1
            return df

    @staticmethod
    def filter_data(df: pd.DataFrame, min_count_rating: int) -> pd.DataFrame:
        # 評価値の数が多いユーザーのみを抽出
        user_counts = df["user_id"].value_counts()
        user_counts = user_counts[user_counts >= min_count_rating].index.tolist()
        df = df[df["user_id"].isin(user_counts)].reset_index(drop=True)

        return df

    @staticmethod
    def split_data(df: pd.DataFrame, test_size: float, seed: int) -> pd.DataFrame:
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=seed, stratify=df["user_id"]
        )
        return train_df, test_df
