from dataclasses import dataclass


@dataclass
class DataConfig:
    name: str = "movielens"
    test_size: float = 0.2


@dataclass
class PredictionConfig:
    model: str = "USERCF"  # or "ITEMCF" or "MF"
    neighbors: int = 10
    lr: float = 0.1
    epoch: int = 10


@dataclass
class MyConfig:
    data_cfg: DataConfig = DataConfig()
    train_cfg: PredictionConfig = PredictionConfig()
    seed: int = 42
