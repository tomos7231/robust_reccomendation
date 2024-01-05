from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.paths import DATA_DIR


class DataProcessor:
    def __init__(self, dataset_name: str, test_size: float = 0.2, seed: int = 42):
        self.dataset_name = dataset_name
        self.seed = seed
        self.test_size = test_size
        self.seed = seed

    def run(self) -> None:
        df = self.load_data(self.dataset_name)

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
            return df

    @staticmethod
    def split_data(df: pd.DataFrame, test_size: float, seed: int) -> pd.DataFrame:
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=seed, stratify=df["user_id"]
        )
        return train_df, test_df
