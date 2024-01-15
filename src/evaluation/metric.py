import numpy as np
import pandas as pd
import tqdm


def precision(items_recommended: dict, pred_rating_df: pd.DataFrame) -> float:
    """
    Precisionを計算する関数
    """
    precisions = list()

    for user in tqdm(items_recommended.keys()):
        num_correct = 0

        # ユーザーごとにデータを抽出
        user_df = pred_rating_df[pred_rating_df["user_id"] == user].reset_index(drop=True)

        # 推薦したアイテムのid
        item_ids = items_recommended[user]
        N = len(item_ids)

        # user_dfでitem_idを探し、評価値が4以上かつdata_typeがtrainならnum_correctを1増やす
        for item_id in item_ids:
            pred_data = user_df[user_df["item_id"] == item_id]
            if (pred_data["rating"].iloc[0] >= 4) and (pred_data["data_type"].iloc[0] == "train"):
                num_correct += 1

        precisions.append(num_correct / N)

    return np.mean(precisions)


def diversity(items_recommended: dict) -> int:
    """
    多様性を計算する関数
    """
    # items_recommendedに含まれる全てのitem_idの種類数
    id_recommended = set()

    for user in items_recommended.keys():
        id_recommended = id_recommended.union(set(items_recommended[user]))

    return len(id_recommended)
