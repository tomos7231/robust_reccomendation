import hydra

from config.config import MyConfig
from src.preprocess.load_data import DataProcessor


@hydra.main(version_base=None, config_path="config/", config_name="config")
def main(cfg: MyConfig):
    train_df, test_df = DataProcessor(cfg.data.name, cfg.data.test_size, cfg.seed).run()
    print(train_df.head())


if __name__ == "__main__":
    main()
