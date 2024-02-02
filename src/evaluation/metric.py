import numpy as np
import pandas as pd
from tqdm import tqdm


def calc_precision(
    items_recommended: dict, test_df: pd.DataFrame, N: int, thres_rating: float
) -> float:
    """
    Precisionを計算する関数
    """
    precisions = list()

    for user in tqdm(items_recommended.keys()):
        num_correct = 0

        # ユーザーごとにデータを抽出
        user_df = test_df[test_df["user_id"] == user].reset_index(drop=True)
        # item_idと評価値の辞書
        rating_dict = dict(zip(user_df["item_id"], user_df["rating"]))

        # 推薦したアイテムのid
        item_ids = items_recommended[user]

        # user_dfでitem_idを探し、存在するかつ評価が閾値以上の場合はnum_correctを+1する
        for item_id in item_ids:
            if item_id in rating_dict.keys() and rating_dict[item_id] >= thres_rating:
                num_correct += 1

        precisions.append(num_correct / N)

    return np.mean(precisions)


def calc_recall(items_recommended: dict, test_df: pd.DataFrame, thres_rating: float) -> float:
    """
    Recallを計算する関数
    """
    recalls = list()

    for user in tqdm(items_recommended.keys()):
        num_correct = 0

        # ユーザーごとに評価が閾値以上のアイテムを抽出
        user_df = test_df[
            (test_df["user_id"] == user) & (test_df["rating"] >= thres_rating)
        ].reset_index(drop=True)
        # user_dfのitem_idがどれだけ推薦したアイテムに含まれているかを計算
        num_correct = len(set(user_df["item_id"]).intersection(set(items_recommended[user])))

        recalls.append(num_correct / len(user_df))

    return np.mean(recalls)


def calc_diversity(items_recommended: dict) -> int:
    """
    多様性を計算する関数
    """
    # items_recommendedに含まれる全てのitem_idの種類数
    id_recommended = set()

    for user in items_recommended.keys():
        id_recommended = id_recommended.union(set(items_recommended[user]))

    return len(id_recommended)
