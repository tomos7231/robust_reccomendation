from __future__ import annotations

from abc import ABCMeta, abstractmethod

import pandas as pd
from surprise import SVD, Dataset, KNNBasic, Reader
from tqdm import tqdm


class RecommenderSystem(metaclass=ABCMeta):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        self.train_df = train_df
        self.test_df = test_df
        self.algo = None

        # train_dfのカラムがuser_id, item_id, ratingの順番であることを保証する
        assert list(train_df.columns) == [
            "user_id",
            "item_id",
            "rating",
        ], "train_df columns must be [user_id, item_id, rating]"

    def run(self) -> pd.DataFrame:
        # データセットの作成
        self.trainset, self.testset = self.make_dataset(self.train_df, self.test_df)
        # モデルの作成と学習
        self.build_model()
        self.fit_model()
        # 予測
        prediction_df = self.predict_ratings()
        return prediction_df

    @staticmethod
    def make_dataset(train_df, test_df) -> tuple:
        max_rate = train_df["rating"].max()
        min_rate = train_df["rating"].min()
        rating_scale = (min_rate, max_rate)

        reader = Reader(rating_scale=rating_scale)
        train_surprise_data = Dataset.load_from_df(train_df, reader)
        test_surprise_data = Dataset.load_from_df(test_df, reader)

        trainset = train_surprise_data.build_full_trainset()
        testset = test_surprise_data.build_full_trainset()
        testset = testset.build_testset()

        return trainset, testset

    @abstractmethod
    def build_model(self) -> None:
        raise NotImplementedError

    def fit_model(self) -> None:
        # モデルがNoneでないことを保証
        assert self.algo is not None, "You must build model before predicting ratings"
        self.algo.fit(self.trainset)

    def predict_ratings(self) -> pd.DataFrame:
        # 全ユーザーとアイテムの組み合わせを生成
        all_users = set(self.train_df["user_id"]).union(set(self.test_df["user_id"]))
        all_items = set(self.train_df["item_id"]).union(set(self.test_df["item_id"]))
        all_combinations = [(user, item) for user in all_users for item in all_items]

        # 結果を保存するリスト
        predictions = []

        print("Predicting ratings...")
        for user, item in tqdm(all_combinations):
            # trainsetに存在するか確認
            if not self.train_df[
                (self.train_df["user_id"] == user) & (self.train_df["item_id"] == item)
            ].empty:
                actual_rating = self.train_df[
                    (self.train_df["user_id"] == user) & (self.train_df["item_id"] == item)
                ]["rating"].iloc[0]
                predictions.append([user, item, actual_rating, "train"])
            # testsetに存在するか確認
            elif not self.test_df[
                (self.test_df["user_id"] == user) & (self.test_df["item_id"] == item)
            ].empty:
                predicted_rating = self.algo.predict(user, item).est
                predictions.append([user, item, predicted_rating, "test"])
            else:
                # trainsetとtestsetのどちらにも存在しない場合
                predicted_rating = self.algo.predict(user, item).est
                predictions.append([user, item, predicted_rating, "temp"])

        predictions_df = pd.DataFrame(
            predictions, columns=["user_id", "item_id", "rating", "data_type"]
        )

        return predictions_df


class SVDRecommender(RecommenderSystem):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        n_factors: int,
        n_epochs: int,
        lr_all: float,
        reg_all: float,
    ):
        super().__init__(train_df, test_df)
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


class UserKNNRecommender(RecommenderSystem):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, k: int):
        super().__init__(train_df, test_df)
        self.k = k

    def build_model(self):
        self.algo = KNNBasic(k=self.k, sim_options={"user_based": True})


class ItemKNNRecommender(RecommenderSystem):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, k: int):
        super().__init__(train_df, test_df)
        self.k = k

    def build_model(self):
        self.algo = KNNBasic(k=self.k, sim_options={"user_based": False})
