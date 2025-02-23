import configparser
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Callable, Dict, Any
from flwr.common import log
from logging import INFO

from flwr_core import (
    get_network_generator,
    get_flower_client_generator,
    FlowerRayClient,
    get_paths,
    fit_client_seeded,
)

from common.client_utils import (
    get_model_parameters,
)

from utils import relative_to_absolute_path


class FlwrTrainConfig(dict):
    """
    Params following the naming convention of Flower training config.
    """
    batch_size: int
    epochs: int
    client_learning_rate: float
    weight_decay: float
    num_workers: int
    max_batches: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        super().__setitem__(key, value)


class FlwrTestConfig(dict):
    """
    Params following the naming convention of Flower testing config.
    """
    batch_size: int
    num_workers: int
    max_batches: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        super().__setitem__(key, value)


class TrainerConfig(ABC):
    """
    A class for the configuration of a trainer.
    """
    @abstractmethod
    def from_file(path: str) -> "TrainerConfig":
        pass

class CentralizedTrainerConfig(TrainerConfig):
    """
    A class for the configuration of a centralized trainer.
    """
    batch_sizes: List[int]
    epochs: int

    train_configs: List[FlwrTrainConfig] = None
    test_configs: List[FlwrTestConfig] = None

    def __init__(self, batch_sizes: List[int], epochs: int, train_configs: List[FlwrTrainConfig], test_configs: List[FlwrTestConfig]):
        self.batch_sizes = batch_sizes
        self.epochs = epochs
        self.train_configs = train_configs
        self.test_configs = test_configs

    @staticmethod
    def from_file(path: str) -> "CentralizedTrainerConfig":
        path = relative_to_absolute_path(path)

        config = configparser.ConfigParser()
        config.read(path)

        train_configs = []
        test_configs = []

        batch_sizes = [int(bs) for bs in config["General"]["batch_sizes"].split(",")]
        epochs = int(config["General"]["epochs"])

        for batch_size in batch_sizes:
            train_configs.append(FlwrTrainConfig(
                batch_size=int(batch_size),
                epochs=epochs,
                client_learning_rate=config["Train"]["learning_rate"],
                weight_decay=config["Train"]["weight_decay"],
                num_workers=config["Train"]["num_workers"],
                max_batches=config["Train"]["max_batches"]
            ))
            test_configs.append(FlwrTestConfig(
                batch_size=int(batch_size),
                num_workers=config["Test"]["num_workers"],
                max_batches=config["Test"]["max_batches"]
            ))

        return CentralizedTrainerConfig(batch_sizes=batch_sizes, epochs=epochs, train_configs=train_configs, test_configs=test_configs)



class Trainer(ABC):
    def __init__(self, config: TrainerConfig):
        self.config = config

    @abstractmethod
    def train(self):
        pass


class CentralizedTrainer(Trainer):
    def __init__(self, config: CentralizedTrainerConfig):
        super().__init__(config)
        self.paths = get_paths()

    def train(self) -> List[List[float]]:
        """
        Train the centralized model (by setting the num clients to 1)

        Args:
            batch_sizes (list[int]): List of batch sizes to train the model with.
            epochs (int, optional): Number of epochs to train each setting for. Defaults to 50.
        
        Returns:
            Metrics for each batch size and each epoch
        """
        metrics = []
        for train_config, test_config in zip(self.config.train_configs, self.config.test_configs):
            curr_metrics = self._train_one_experiment(train_config, test_config)
            metrics.append(curr_metrics)
        
        return self._organize_metrics(metrics)

    def _train_one_experiment(self, train_config: FlwrTrainConfig, test_config: FlwrTestConfig) -> Dict[str, Any]:
        network_generator = get_network_generator()
        seed_net = network_generator()
        seed_model_params = get_model_parameters(seed_net)

        # training
        centralized_flower_client_generator: Callable[[str], FlowerRayClient] = (
            get_flower_client_generator(network_generator, self.paths["centralized_partition"])
        )
        centralized_flower_client = centralized_flower_client_generator(str(0))


        trained_params, num_examples, train_metrics = fit_client_seeded(
            centralized_flower_client, params=seed_model_params, conf=train_config
        )
        log(INFO, "Train Metrics = %s", train_metrics)


        # testing
        loss, num_examples, test_metrics = centralized_flower_client.evaluate(
            parameters=trained_params, config=test_config
        )
        log(INFO, "Loss = %s; Test Metrics = %s", loss, test_metrics)

        return {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics
        }
    
    def _organize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        
        
        return {
            "batch_sizes": self.config.batch_sizes,
            "train_loss": [m["train_metrics"]["train_loss"] for m in metrics],
            "noise_scale": [m["train_metrics"]["noise_scale"] for m in metrics],
            "test_accuracy": [m["test_metrics"]["local_accuracy"] for m in metrics]
        }


if __name__ == "__main__":
    config = CentralizedTrainerConfig.from_file("centralized_config.ini")
    trainer = CentralizedTrainer(config)
    metrics = trainer.train()
    print(metrics)
