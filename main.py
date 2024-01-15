import logging

import hydra

from config.config import MyConfig
from src.evaluation.evaluate import evaluate
from src.optimization.make_covariance_matrix import make_covariance_matrix
from src.optimization.optimize import optimize
from src.prediction.predict import predict_ratings
from src.preprocess.load_data import DataProcessor

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config/", config_name="config")
def main(cfg: MyConfig):
    # データの読み込み
    train_df, test_df = DataProcessor(cfg.data.name, cfg.data.test_size, cfg.seed).run()

    # 評価値予測
    print("predict ratings...")
    predict_ratings(cfg, train_df, test_df, logger, cfg.prediction.model)

    # 共分散行列の推定
    print("make covariance matrix...")
    make_covariance_matrix(cfg.optimization.delta, cfg.optimization.estimator)

    # 最適化問題を解く
    print("optimize recommended item...")
    optimize(
        cfg.optimization.alpha,
        cfg.optimization.gamma_mu,
        cfg.optimization.gamma_sigma,
        cfg.optimization.c_mu,
        cfg.optimization.c_sigma,
        cfg.optimization.N,
    )

    # 評価指標を計算
    print("evaluate recommended item...")
    evaluate(logger)


if __name__ == "__main__":
    main()
