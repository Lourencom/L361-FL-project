import random
from pathlib import Path
import tarfile
from typing import Any
from logging import INFO, DEBUG
from collections import defaultdict, OrderedDict
from collections.abc import Sequence, Callable
import numbers
import time
from typing import Optional, Tuple, List
import numpy as np
import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from enum import IntEnum
import flwr
from flwr.server import History, ServerConfig
from flwr.server.server_returns_parameters import ReturnParametersServer
from flwr.server.strategy import FedAvgM as FedAvg, Strategy
from flwr.common import log, NDArrays, Scalar, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.client.client import Client
import timeit
import matplotlib.pyplot as plt

from .common.client import FlowerClient
from .common.client_utils import (
    Net,
    load_femnist_dataset,
    get_network_generator_cnn as get_network_generator,
    train_femnist,
    test_femnist,
    save_history,
    set_model_parameters,
    get_model_parameters,
    get_device,
)
from .utils import get_git_root
from .estimate import (
    collect_gradients,
    compute_noise_scale_from_gradients,
)


# Add new seeds here for easy autocomplete
class Seeds(IntEnum):
    """Seeds for reproducibility."""

    DEFAULT = 1337


def set_all_seeds():
    np.random.seed(Seeds.DEFAULT)
    random.seed(Seeds.DEFAULT)
    torch.manual_seed(Seeds.DEFAULT)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_paths():
    home_dir = Path(get_git_root())
    dataset_dir: Path = home_dir / "femnist"
    data_dir: Path = dataset_dir / "data"
    centralized_partition: Path = dataset_dir / "client_data_mappings" / "centralized"
    centralized_mapping: Path = dataset_dir / "client_data_mappings" / "centralized" / "0"
    federated_partition: Path = dataset_dir / "client_data_mappings" / "fed_natural"
    iid_partition: Path = dataset_dir / "client_data_mappings" / "fed_iid"
    
    paths = {
        "home_dir": home_dir,
        "dataset_dir": dataset_dir,
        "data_dir": data_dir,
        "centralized_partition": centralized_partition,
        "centralized_mapping": centralized_mapping,
        "federated_partition": federated_partition,
        "iid_partition": iid_partition
    }
    return paths


def decompress_dataset(paths: dict):
    if not paths["dataset_dir"].exists():
        with tarfile.open(paths["dataset_dir"] / "femnist.tar.gz", "r:gz") as tar:
            tar.extractall(path=paths["dataset_dir"])
        log(INFO, "Dataset extracted in %s", paths["dataset_dir"])


class FlowerRayClient(flwr.client.NumPyClient):
    """Flower client for the FEMNIST dataset."""

    def __init__(
        self,
        cid: int,
        partition_dir: Path,
        model_generator: Callable[[], Module],
    ) -> None:
        """Init the client with its unique id and the folder to load data from.

        Parameters:
            cid (int): Unique client id for a client used to map it to its data
                partition
            partition_dir (Path): The directory containing data for each
                client/client id
            model_generator (Callable[[], Module]): The model generator function
        
        """
        self.cid = cid
        log(INFO, "cid: %s", self.cid)
        self.partition_dir = partition_dir
        self.device = str(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.model_generator: Callable[[], Module] = model_generator
        self.properties: dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

    def set_parameters(self, parameters: NDArrays) -> Module:
        """Load weights inside the network."""
        net = self.model_generator()
        return set_model_parameters(net, parameters)

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return weights from a given model.

        If no model is passed, then a local model is created.
        This can be used to initialise a model in the
        server.
        The config param is not used but is mandatory in Flower.

        """
        net = self.model_generator()
        return get_model_parameters(net)

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict]:
        """Receive and train a model on the local client data."""
        
        # Only create model right before training/testing
        # To lower memory usage when idle
        net = self.set_parameters(parameters)
        net.to(self.device)

        train_loader: DataLoader = self._create_data_loader(config, name="train")
        train_loss, training_time = self._train(net, train_loader=train_loader, config=config)

        # Compute gradients and noise scale
        grad_vectors = collect_gradients(net, train_loader, self.device, torch.nn.CrossEntropyLoss(), 5)
        # Compute local noise scale (Bsimple) on this client.
        local_noise_scale = compute_noise_scale_from_gradients(grad_vectors)
        
        # Convert the gradient tensor to a serializable format (numpy array)
        G_local = torch.stack(grad_vectors).flatten()
        G_local_list = G_local.cpu().numpy().tolist()
        G_local_dict = {str(i): G_local_list[i] for i in range(len(G_local_list)) if i % 20 == 0}
        
        # Calculate actual samples processed considering max_batches
        max_batches = config.get("max_batches", float("inf"))
        actual_batches = min(len(train_loader), max_batches)
        samples_processed = actual_batches * config["batch_size"]

        metrics_dict = {
            "train_loss": train_loss, 
            "noise_scale": local_noise_scale,
            "training_time": training_time,
            "samples_processed": samples_processed,
            "actual_batches": actual_batches
        }

         # join metrics dict and G_local_dict
        if config.get("return_params", False):
            return_dict = {**metrics_dict, **G_local_dict}
            return get_model_parameters(net), len(train_loader), return_dict
        else:
            return get_model_parameters(net), len(train_loader), metrics_dict

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, int, dict]:
        """Receive and test a model on the local client data."""
        net = self.set_parameters(parameters)
        net.to(self.device)

        test_loader: DataLoader = self._create_data_loader(config, name="test")
        loss, accuracy = self._test(net, test_loader=test_loader, config=config)
        return loss, len(test_loader), {"local_accuracy": accuracy}

    def _create_data_loader(self, config: dict[str, Scalar], name: str) -> DataLoader:
        """Create the data loader using the specified config parameters."""
        batch_size = int(config["batch_size"])
        num_workers = int(config["num_workers"])
        dataset = self._load_dataset(name)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=(name == "train"),
        )

    def _load_dataset(self, name: str) -> Dataset:
        data_dir = get_paths()["data_dir"]
        full_file: Path = self.partition_dir / str(self.cid)
        return load_femnist_dataset(
            mapping=full_file,
            name=name,
            data_dir=data_dir,
        )

    def _train(
        self, net: Module, train_loader: DataLoader, config: dict[str, Scalar]
    ) -> float:
        return train_femnist(
            net=net,
            train_loader=train_loader,
            epochs=int(config["epochs"]),
            device=self.device,
            optimizer=torch.optim.AdamW(
                net.parameters(),
                lr=float(config["client_learning_rate"]),
                weight_decay=float(config["weight_decay"]),
            ),
            criterion=torch.nn.CrossEntropyLoss(),
            max_batches=None if "max_batches" not in config else int(config["max_batches"]),
            return_total_time=True,
        )

    def _test(
        self, net: Module, test_loader: DataLoader, config: dict[str, Scalar]
    ) -> tuple[float, float]:
        return test_femnist(
            net=net,
            test_loader=test_loader,
            device=self.device,
            criterion=torch.nn.CrossEntropyLoss(),
            max_batches=None if "max_batches" not in config else int(config["max_batches"]),
        )

    def get_properties(self, config: dict[str, Scalar]) -> dict[str, Scalar]:
        """Return properties for this client."""
        return self.properties

    def get_train_set_size(self) -> int:
        """Return the client train set size."""
        return len(self._load_dataset("train"))  # type: ignore[reportArgumentType]

    def get_test_set_size(self) -> int:
        """Return the client test set size."""
        return len(self._load_dataset("test"))  # type: ignore[reportArgumentType]


class TargetAccuracyServer(ReturnParametersServer):
    """Server that runs until reaching a target accuracy."""

    def __init__(
        self,
        *args,
        target_accuracy: float = 0.60,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.target_accuracy = target_accuracy

    def fit(self, num_rounds: int, timeout: Optional[float]) -> Tuple[List[Tuple[int, NDArrays]], History]:
        """Run federated averaging until target accuracy is reached."""
        history = History()
        current_round = 0

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        self.return_params.append((0, parameters_to_ndarrays(self.parameters)))
        
        # Initial evaluation
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0], res[1])
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning until target accuracy or max rounds
        log(INFO, f"FL starting - Target accuracy: {self.target_accuracy}")
        start_time = timeit.default_timer()

        while True:
            current_round += 1
            
            # Train model
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )
                if self.return_all_parameters:
                    self.return_params.append((current_round, parameters_to_ndarrays(self.parameters)))

            # Evaluate model
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                current_accuracy = metrics_cen.get("accuracy", 0.0)
                log(
                    INFO,
                    "fit progress: (round %s, accuracy %s, loss %s, time %s)",
                    current_round,
                    current_accuracy,
                    loss_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

                # Check if target accuracy reached
                if current_accuracy >= self.target_accuracy:
                    log(INFO, f"Target accuracy {self.target_accuracy} reached in round {current_round}")
                    break

            # Evaluate on sample clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(server_round=current_round, loss=loss_fed)
                    history.add_metrics_distributed(server_round=current_round, metrics=evaluate_metrics_fed)

        if not self.return_all_parameters:
            self.return_params.append((current_round, parameters_to_ndarrays(self.parameters)))

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s seconds after %s rounds", elapsed, current_round)
        return self.return_params, history


def fit_client_seeded(
    client: FlowerRayClient,
    params: NDArrays,
    conf: dict[str, Any],
    seed: Seeds = Seeds.DEFAULT,
    **kwargs: Any,
) -> tuple[NDArrays, int, dict]:
    """Wrap to always seed client training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return client.fit(params, conf, **kwargs)


def get_flower_client_generator(
    model_generator: Callable[[], Module],
    partition_dir: Path,
    mapping_fn: Callable[[int], int] | None = None,
) -> Callable[[str], FlowerRayClient]:
    """Wrap the client instance generator.

    A mapping function could be used for filtering/ordering clients.

    Parameters
    ----------
        model_generator (Callable[[], Module]): model generator function.
        partition_dir (Path): directory containing the partition.
        mapping_fn (Optional[Callable[[int], int]]): function mapping sorted/filtered
            ids to real cid.
    """

    def client_fn(cid: str) -> FlowerRayClient:
        """Create a single client instance given the client id `cid`."""
        return FlowerRayClient(
            cid=mapping_fn(int(cid)) if mapping_fn is not None else int(cid),
            partition_dir=partition_dir,
            model_generator=model_generator,
        )

    return client_fn


def sample_random_clients(
    total_clients: int,
    filter_less: int,
    cid_client_generator: Callable[[str], FlowerClient],
    seed: int | None = Seeds.DEFAULT,
    max_clients: int = 3229,
) -> Sequence[int]:
    """Sample randomly clients.

    A filter on the client train set size is performed.

    Parameters
    ----------
        total_clients (int): total number of clients to sample.
        filter_less (int): max number of train samples for which the client is
            **discarded**.
    """
    if seed is not None:
        random.seed(seed)
    list_of_ids = []
    while len(list_of_ids) < total_clients:
        current_id = random.randint(0, max_clients)
        if (
            cid_client_generator(str(current_id)).get_train_set_size()
            > filter_less
        ):
            list_of_ids.append(current_id)
    return list_of_ids


def get_federated_evaluation_function(
    batch_size: int,
    num_workers: int,
    model_generator: Callable[[], Module],
    criterion: Module,
    max_batches: int | None = None,
) -> Callable[[int, NDArrays, dict[str, Any]], tuple[float, dict[str, Scalar]]]:
    """Wrap the external federated evaluation function.

    It provides the external federated evaluation function with some
    parameters for the dataloader, the model generator function, and
    the criterion used in the evaluation.

    Parameters
    ----------
        batch_size (int): batch size of the test set to use.
        num_workers (int): correspond to `num_workers` param in the Dataloader object.
        model_generator (Callable[[], Module]):  model generator function.
        criterion (Module): PyTorch Module containing the criterion for evaluating the
        model.

    Returns
    ----------
        External federated evaluation function.
    """

    def federated_evaluation_function(
        server_round: int,
        parameters: NDArrays,
        fed_eval_config: dict[
            str, Any
        ],  # mandatory argument, even if it's not being used
    ) -> tuple[float, dict[str, Scalar]]:
        """Evaluate federated model on the server.

        It uses the centralized val set for sake of simplicity.

        Parameters
        ----------
            server_round (int): current federated round.
            parameters (NDArrays): current model parameters.
            fed_eval_config (dict[str, Any]): mandatory argument in Flower, can contain
                some configuration info

        Returns
        -------
            tuple[float, dict[str, Scalar]]: evaluation results
        """
        device: str = get_device() # FIXME
        net: Module = set_model_parameters(model_generator(), parameters)
        net.to(device)

        full_file: Path = get_paths()["centralized_mapping"] # FIXME
        dataset: Dataset = load_femnist_dataset(get_paths()["data_dir"], full_file, "val") # FIXME

        valid_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        loss, acc = test_femnist(
            net=net,
            test_loader=valid_loader,
            device=device,
            criterion=criterion,
            max_batches=max_batches,
        )
        return loss, {"accuracy": acc}

    return federated_evaluation_function


def create_iid_partition(paths: dict, num_clients: int = 10, seed: int = 42):
    """Create IID partition by randomly distributing data among clients."""
    import pandas as pd
    import numpy as np
    import shutil
    
    # remove directory if it exists and has num_clients clients
    if paths["iid_partition"].exists():
        curr_num_clients = len(list(paths["iid_partition"].iterdir()))
        print(f"Current number of clients in iid partition: {curr_num_clients}, expected: {num_clients}")
        if curr_num_clients == num_clients:
            print("IID partition already exists and has the correct number of clients")
            return
        else:
            shutil.rmtree(paths["iid_partition"])
    
    # Create directory if it doesn't exist
    iid_dir = paths["iid_partition"]
    iid_dir.mkdir(parents=True, exist_ok=True)
    
    # Read all training and test data from centralized partition
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    
    # Read and combine all data from centralized partition
    centralized_dir = paths["centralized_partition"]
    for client_dir in centralized_dir.iterdir():
        if client_dir.is_dir():
            train_file = client_dir / "train.csv"
            test_file = client_dir / "test.csv"
            
            if train_file.exists():
                client_train = pd.read_csv(train_file)
                train_data = pd.concat([train_data, client_train], ignore_index=True)
            
            if test_file.exists():
                client_test = pd.read_csv(test_file)
                test_data = pd.concat([test_data, client_test], ignore_index=True)
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Shuffle the data
    train_data = train_data.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_data = test_data.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Calculate samples per client
    train_samples_per_client = len(train_data) // num_clients
    test_samples_per_client = len(test_data) // num_clients
    
    # Distribute data to clients
    for client_id in range(num_clients):
        print(f"Distributing data to client {client_id}")
        client_dir = iid_dir / str(client_id)
        client_dir.mkdir(exist_ok=True)
        
        # Get client's portion of data
        train_start = client_id * train_samples_per_client
        train_end = train_start + train_samples_per_client if client_id < num_clients - 1 else len(train_data)
        
        test_start = client_id * test_samples_per_client
        test_end = test_start + test_samples_per_client if client_id < num_clients - 1 else len(test_data)
        
        # Assign client_id to the data
        client_train_data = train_data.iloc[train_start:train_end].copy()
        client_test_data = test_data.iloc[test_start:test_end].copy()
        
        client_train_data['client_id'] = client_id
        client_test_data['client_id'] = client_id
        
        # Save train and test data for this client
        client_train_data.to_csv(client_dir / "train.csv", index=False)
        client_test_data.to_csv(client_dir / "test.csv", index=False)
