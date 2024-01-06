import logging

import hydra

from config.config import MyConfig
from src.optimization.estimator import DiagonalEstimator, InputationEstimator
from src.optimization.make_covariance_matrix import make_covariance_matrix
from src.paths import RESULT_DIR
from src.prediction.predict import predict_ratings
from src.preprocess.load_data import DataProcessor

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config/", config_name="config")
def main(cfg: MyConfig):
    # データの読み込み
    train_df, test_df = DataProcessor(cfg.data.name, cfg.data.test_size, cfg.seed).run()

    # 評価値予測
    predict_ratings(cfg, train_df, test_df, logger, cfg.prediction.model)

    # 共分散行列の推定
    make_covariance_matrix(cfg.optimization.delta, cfg.name, cfg.optimization.estimator)


if __name__ == "__main__":
    main()
