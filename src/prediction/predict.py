from __future__ import annotations

import pandas as pd

from src.paths import RESULT_DIR
from src.prediction.model import ItemKNNRecommender, SVDRecommender, UserKNNRecommender


def predict_ratings(cfg: dict, train_df: pd.DataFrame, test_df: pd.DataFrame, model_name: str):
    if model_name == "MF":
        model = SVDRecommender(
            train_df,
            test_df,
            cfg.prediction.n_factors,
            cfg.prediction.n_epochs,
            cfg.prediction.lr_all,
            cfg.prediction.reg_all,
        )
    elif model_name == "USERCF":
        model = UserKNNRecommender(train_df, test_df, cfg.prediction.k)
    elif model_name == "ITEMCF":
        model = ItemKNNRecommender(train_df, test_df, cfg.prediction.k)
    else:
        raise Exception("Unknown model name: {}".format(model_name))

    pred_all_df = model.run()

    print(pred_all_df.head(10))

    # 保存
    pred_all_df.to_csv(RESULT_DIR / cfg.name / "pred_rating.csv", index=False)
