from dataclasses import dataclass


@dataclass
class DataConfig:
    name: str = "movielens"
    test_size: float = 0.2


@dataclass
class PredictionConfig:
    model: str = "USERCF"  # or "ITEMCF" or "MF"
    k: int = 5
    n_factors: int = 100
    n_epochs: int = 20
    lr_all: float = 0.01
    reg_all: float = 0.2


@dataclass
class OptimizationConfig:
    delta: float = 0.5
    estimator: str = "DIAG"


@dataclass
class MyConfig:
    data_cfg: DataConfig = DataConfig()
    train_cfg: PredictionConfig = PredictionConfig()
    optim_cfg: OptimizationConfig = OptimizationConfig()
    seed: int = 42
    name: str = "default"
