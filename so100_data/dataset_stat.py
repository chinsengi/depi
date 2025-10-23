import logging
from pprint import pformat

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    init_logging,
)
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig


@parser.wrap()
def main(cfg: TrainPipelineConfig):
    cfg.validate()

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Reinitialize logging to only log on main process
    init_logging()
    logging.info(pformat(cfg.to_dict()))

    # Log accelerator configuration
    logging.info("Creating dataset")
    dataset = make_dataset(cfg)
    print(dataset.meta.stats)


if __name__ == "__main__":
    main()
