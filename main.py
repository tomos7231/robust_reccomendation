import logging

import hydra

from config.config import MyConfig
from src.paths import RESULT_DIR
from src.prediction.predict import predict_ratings
from src.preprocess.load_data import DataProcessor

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config/", config_name="config")
def main(cfg: MyConfig):
    # dir
    OUTPUT_DIR = RESULT_DIR / cfg.data.name
    logger.info(f"OUTPUT_DIR: {OUTPUT_DIR}")

    train_df, test_df = DataProcessor(cfg.data.name, cfg.data.test_size, cfg.seed).run()
    # uidでソートしてprint
    print(train_df.sort_values(["user_id", "item_id"]).head(10))

    predict_ratings(cfg, train_df, test_df, cfg.prediction.model)


if __name__ == "__main__":
    main()
