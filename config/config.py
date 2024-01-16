from dataclasses import dataclass


@dataclass
class DataConfig:
    name: str = "movielens"
    test_size: float = 0.2
    min_count_rating: int = 50
    thres_rating: float = 4.0


@dataclass
class PredictionConfig:
    model: str = "MF"  # or "ITEMCF" or "MF"
    k: int = 5
    n_factors: int = 100
    n_epochs: int = 20
    lr_all: float = 0.01
    reg_all: float = 0.2


@dataclass
class OptimizationConfig:
    estimator: str = "DIAG"
    delta: float = 0.5
    n_candidate: int = 50
    alpha: float = 0.2
    gamma_mu: int = 5
    gamma_sigma: int = 50
    c_mu: float = 1
    c_sigma: float = 1
    N: int = 10


@dataclass
class MyConfig:
    data_cfg: DataConfig = DataConfig()
    train_cfg: PredictionConfig = PredictionConfig()
    optim_cfg: OptimizationConfig = OptimizationConfig()
    seed: int = 42
    name: str = "default"
