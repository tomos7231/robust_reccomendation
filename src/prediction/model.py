from __future__ import annotations

from abc import ABCMeta, abstractmethod

import pandas as pd
from surprise import SVD, Dataset, KNNBasic, Reader
from tqdm import tqdm


class RecommenderSystem(metaclass=ABCMeta):
    def __init__(self, train_df: pd.DataFrame):
        self.train_df = train_df

        # train_dfのカラムがuser_id, item_id, ratingの順番であることを保証する
        assert list(train_df.columns) == [
            "user_id",
            "item_id",
            "rating",
        ], "train_df columns must be [user_id, item_id, rating]"

        max_rate = train_df["rating"].max()
        min_rate = train_df["rating"].min()
        rating_scale = (min_rate, max_rate)

        self.reader = Reader(rating_scale=rating_scale)
        self.data = Dataset.load_from_df(train_df, self.reader)
        self.trainset = self.data.build_full_trainset()
        self.algo = None

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    def predict_ratings(self) -> list:
        # モデルをビルドしていない場合はエラーを投げる
        assert self.algo is not None, "You must build model before predicting ratings"

        print("Predicting ratings...")

        trained_pairs = set(
            self.train_df[["user_id", "item_id"]].itertuples(index=False, name=None)
        )
        predictions = []

        for user in tqdm(self.trainset.all_users()):
            for item in self.trainset.all_items():
                if (user, item) not in trained_pairs:
                    pred = self.algo.predict(user, item)
                    predictions.extend([(pred.uid, pred.iid, pred.est)])

        return predictions


class SVDRecommender(RecommenderSystem):
    def __init__(
        self, train_df: pd.DataFrame, n_factors: int, n_epochs: int, lr_all: float, reg_all: float
    ):
        super().__init__(train_df)
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all

    def build_model(self):
        self.algo = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
        )
        self.algo.fit(self.trainset)


class UserKNNRecommender(RecommenderSystem):
    def __init__(self, train_df: pd.DataFrame, k: int):
        super().__init__(train_df)
        self.k = k

    def build_model(self):
        self.algo = KNNBasic(k=self.k, sim_options={"user_based": True})
        self.algo.fit(self.trainset)


class ItemKNNRecommender(RecommenderSystem):
    def __init__(self, train_df: pd.DataFrame, k: int):
        super().__init__(train_df)
        self.k = k

    def build_model(self):
        self.algo = KNNBasic(k=self.k, sim_options={"user_based": False})
        self.algo.fit(self.trainset)
