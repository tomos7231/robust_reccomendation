from __future__ import annotations

import pandas as pd

from src.prediction.model import (
    ItemKNNRecommender,
    RecommenderSystem,
    SVDRecommender,
    UserKNNRecommender,
)


def predict_ratings(
    cfg: dict, train_df: pd.DataFrame, model_name: str
) -> list[tuple[str, str, float]]:  # noqa: F821
    if model_name == "MF":
        model = SVDRecommender(
            train_df,
            cfg.prediction.n_factors,
            cfg.prediction.n_epochs,
            cfg.prediction.lr_all,
            cfg.prediction.reg_all,
        )
    elif model_name == "USERCF":
        model = UserKNNRecommender(train_df, cfg.prediction.k)
    elif model_name == "ITEMCF":
        model = ItemKNNRecommender(train_df, cfg.prediction.k)
    else:
        raise Exception("Unknown model name: {}".format(model_name))

    model.build_model()
    rating = model.predict_ratings()
    return rating
