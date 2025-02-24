import torch
import torch.nn as nn
import numpy as np
import random
from logging import INFO

from typing import Callable

import flwr
from flwr.server import History, ServerConfig
from flwr.server.strategy import Strategy, FedAvgM as FedAvg
from flwr.common import log, NDArrays, Scalar, Parameters, ndarrays_to_parameters
from flwr.client.client import Client

from src.flwr_core import Seeds, get_paths

from src.common.client_utils import (
    save_history,
)

from src.estimate import (
    compute_noise_scale_from_gradients,
    collect_gradients,
)


def centralized_experiment(centralized_train_cfg, centralized_test_cfg, train_loader, test_loader, device, network):
    model = network.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=centralized_train_cfg["client_learning_rate"],
        weight_decay=centralized_train_cfg["weight_decay"]
        )
    criterion = nn.CrossEntropyLoss()

    epoch_accuracies = []
    epoch_losses = []
    epoch_noise_scales = []

    for epoch in range(centralized_train_cfg["epochs"]):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            if "max_batches" in centralized_train_cfg and batch_idx >= centralized_train_cfg["max_batches"]:
                break
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
        running_loss /= len(train_loader.dataset)
        epoch_losses.append(running_loss)

        # collect gradients over a few mini-batches
        grad_vectors = collect_gradients(model, train_loader, device, criterion, 5) 
        noise_scale = compute_noise_scale_from_gradients(grad_vectors)
        epoch_noise_scales.append(noise_scale)

        # Evaluate the trained model
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if "max_batches" in centralized_test_cfg and batch_idx >= centralized_test_cfg["max_batches"]:
                    break
                data, target = data.to(device), target.to(device)
                output = model(data)
                preds = output.argmax(dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)
        accuracy = correct / total
        epoch_accuracies.append(accuracy)

        log(INFO, f"Epoch {epoch+1}/{centralized_train_cfg['epochs']}, Loss: {running_loss:.4f}, "
              f"Noise scale: {noise_scale:.4e}, Accuracy: {accuracy*100:.2f}%")
    
    return {
        "accuracies": epoch_accuracies,
        "losses": epoch_losses,
        "noise_scales": epoch_noise_scales,
    }



def start_seeded_simulation(
    client_fn: Callable[[str], Client],
    num_clients: int,
    config: ServerConfig,
    strategy: Strategy,
    name: str,
    return_all_parameters: bool = False,
    seed: int = Seeds.DEFAULT,
    iteration: int = 0,
) -> tuple[list[tuple[int, NDArrays]], History]:
    """Wrap to seed client selection."""
    np.random.seed(seed ^ iteration)
    torch.manual_seed(seed ^ iteration)
    random.seed(seed ^ iteration)
    parameter_list, hist = flwr.simulation.start_simulation_no_ray(
        client_fn=client_fn,
        num_clients=num_clients,
        client_resources={},
        config=config,
        strategy=strategy,
    )
    save_history(get_paths()["home_dir"], hist, name) # FIXME
    return parameter_list, hist


def run_simulation(
    num_rounds: int,
    num_total_clients: int,
    num_clients_per_round: int,
    num_evaluate_clients: int,
    min_available_clients: int,
    min_fit_clients: int,
    min_evaluate_clients: int,
    evaluate_fn: (Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None] | None),
    on_fit_config_fn: Callable[[int], dict[str, Scalar]],
    on_evaluate_config_fn: Callable[[int], dict[str, Scalar]],
    initial_parameters: Parameters,
    fit_metrics_aggregation_fn: Callable | None,
    evaluate_metrics_aggregation_fn: Callable | None,
    federated_client_generator: Callable[[str], flwr.client.NumPyClient],
    server_learning_rate: float = 1.0,
    server_momentum: float = 0.0,
    accept_failures: bool = False,
) -> tuple[list[tuple[int, NDArrays]], History]:
    """Run a federated simulation using Flower."""
    log(INFO, "FL will execute for %s rounds", num_rounds)

    # Percentage of clients used for train/eval
    fraction_fit: float = float(num_clients_per_round) / num_total_clients
    fraction_evaluate: float = float(num_evaluate_clients) / num_total_clients

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_parameters,
        accept_failures=accept_failures,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        server_learning_rate=server_learning_rate,
        server_momentum=server_momentum,
    )
    # resetting the seed for the random selection of clients
    # this way the list of clients trained is guaranteed to be always the same

    cfg = ServerConfig(num_rounds)

    def simulator_client_generator(cid: str) -> Client:
        return federated_client_generator(cid).to_client()

    parameters_for_each_round, hist = start_seeded_simulation(
        client_fn=simulator_client_generator,
        num_clients=num_total_clients,
        config=cfg,
        strategy=strategy,
        name="fedavg",
        return_all_parameters=True,
        seed=Seeds.DEFAULT,
    )
    return parameters_for_each_round, hist
