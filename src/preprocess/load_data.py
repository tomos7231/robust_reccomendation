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
        train_df, test_df = self.split_data(df, self.test_size, self.dataset_name, self.seed)
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

        elif dataset_name == "r3":
            df = pd.read_csv(DATA_DIR / "r3.csv")
            # # user_id and item_id start from 0
            # df["user_id"] -= 1
            # df["item_id"] -= 1
            return df

        elif dataset_name == "book":
            df = pd.read_csv(DATA_DIR / "book.csv")
            return df

    @staticmethod
    def filter_data(df: pd.DataFrame, min_count_rating: int) -> pd.DataFrame:
        # 評価値の数が多いユーザーのみを抽出
        user_counts = df["user_id"].value_counts()
        user_counts = user_counts[user_counts >= min_count_rating].index.tolist()
        df = df[df["user_id"].isin(user_counts)].reset_index(drop=True)

        return df

    @staticmethod
    def split_data(
        df: pd.DataFrame, test_size: float, dataset_name: str, seed: int
    ) -> pd.DataFrame:
        if dataset_name == "movielens":
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=seed, stratify=df["user_id"]
            )
            return train_df, test_df

        elif dataset_name == "r3":
            # dataのカラムで分割
            train_df = df[df["data"] == "train"].drop("data", axis=1).reset_index(drop=True)
            test_df = df[df["data"] == "test"].drop("data", axis=1).reset_index(drop=True)
            return train_df, test_df

        elif dataset_name == "book":
            # # まずuser_idを1000個サンプリングし、そのuser_idに対応するデータを抽出
            # user_ids = df["user_id"].unique()
            # np.random.seed(seed)
            # sampled_user_ids = np.random.choice(user_ids, 1000, replace=False)
            # _df = df[df["user_id"].isin(sampled_user_ids)].reset_index(drop=True)
            # train_test_split
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=seed, stratify=df["user_id"]
            )
            return train_df, test_df
