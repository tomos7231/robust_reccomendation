from src.prediction.model import (
    ItemKNNRecommender,
    RecommenderSystem,
    SVDRecommender,
    UserKNNRecommender,
)


def predict_ratings(train_df, model_name):
    if model_name == "MF":
        model = SVDRecommender(train_df)
    elif model_name == "USERCF":
        model = UserKNNRecommender(train_df)
    elif model_name == "ITEMCF":
        model = ItemKNNRecommender(train_df)
    else:
        raise Exception("Unknown model name: {}".format(model_name))

    model.build_model()
    rating = model.predict_ratings()
    return rating
