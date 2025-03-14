import torch
import torch.nn as nn
import numpy as np
import random
from logging import INFO
import time
from typing import Callable, Optional

import flwr
from flwr.server import History, ServerConfig
from flwr.server.strategy import Strategy, FedAvgM as FedAvg
from flwr.common import log, NDArrays, Scalar, Parameters, ndarrays_to_parameters
from flwr.client.client import Client
from flwr.server.server_returns_parameters import ReturnParametersServer
from flwr.server.client_manager import ClientManager, SimpleClientManager

from src.flwr_core import Seeds, get_paths, TargetAccuracyServer

from src.common.client_utils import (
    save_history,
)

from src.estimate import (
    compute_noise_scale_from_gradients,
    collect_gradients,
    collect_accumulated_gradients,
)


def centralized_experiment(centralized_train_cfg, centralized_test_cfg, train_loader, test_loader, device, network):
    """
    Theoretically we dont care about the losses or accuracies, but for now we keep them.
    """
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
    epoch_times = [] # time per epoch
    epoch_compute_budgets = [] # cumulative number of samples processed

    #for epoch in range(centralized_train_cfg["epochs"]):
    epoch = 0
    total_time = 0.0
    while True: 
        model.train()
        running_loss = 0.0
        cumulative_samples = 0

        

        val_epoch_interval = 50

        finished = False
        epoch_time = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            if "max_batches" in centralized_train_cfg and batch_idx >= centralized_train_cfg["max_batches"]:
                break
            
            data, target = data.to(device), target.to(device)
            start_time = time.time()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            end_time = time.time()
            total_time += end_time - start_time
            epoch_time += end_time - start_time
            cumulative_samples += data.size(0)
            
            running_loss += loss.item() * data.size(0)

            if batch_idx % val_epoch_interval == 0:
                # Evaluate the trained model
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(train_loader):
                        if "max_batches" in centralized_test_cfg and batch_idx >= centralized_test_cfg["max_batches"]:
                            break
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        preds = output.argmax(dim=1)
                        correct += (preds == target).sum().item()
                        total += target.size(0)

                acc = correct / total

                if acc > centralized_test_cfg["target_accuracy"]:
                    log(INFO, "Epoch finished early, target accuracy reached")
                    finished = True
                    break
            
        if finished:
            break

        
         # Evaluate the trained model
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):
                if "max_batches" in centralized_test_cfg and batch_idx >= centralized_test_cfg["max_batches"]:
                    break
                data, target = data.to(device), target.to(device)
                output = model(data)
                preds = output.argmax(dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)
        accuracy = correct / total
        epoch_accuracies.append(accuracy)

        
        epoch_compute_budgets.append(cumulative_samples)

        running_loss /= len(train_loader.dataset)
        epoch_losses.append(running_loss)

        # collect gradients over a few mini-batches
        grad_vectors = collect_gradients(model, train_loader, device, criterion, 5) 
        noise_scale = compute_noise_scale_from_gradients(grad_vectors)
        epoch_noise_scales.append(noise_scale)


        log(INFO, f"Epoch {epoch+1}/{centralized_train_cfg['epochs']}, Loss: {running_loss:.4f}, "
              f"Noise scale: {noise_scale:.4e}, Accuracy: {accuracy*100:.2f}%, Epoch time: {epoch_time:.2f}s")
        
        if accuracy > centralized_test_cfg["target_accuracy"]:
            break
        
        epoch += 1

    return {
        "accuracies": epoch_accuracies,
        "losses": epoch_losses,
        "noise_scales": epoch_noise_scales,
        "training_time": total_time,
        "compute_cost": epoch_compute_budgets,
    }





def centralized_critical_bs_estimation(centralized_train_cfg, centralized_test_cfg, train_loader, test_loader, device, network, accumulation_steps=8, log = False):
    """
    Theoretically we dont care about the losses or accuracies, but for now we keep them.
    """
    model = network.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=centralized_train_cfg["client_learning_rate"],
        weight_decay=centralized_train_cfg["weight_decay"]
        )
    criterion = nn.CrossEntropyLoss()

    B_small = centralized_train_cfg["batch_size"]
    B_big = B_small * accumulation_steps

    B_simples = []
    for epoch in range(centralized_train_cfg["epochs"]):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            if "max_batches" in centralized_train_cfg and batch_idx >= centralized_train_cfg["max_batches"]:
                break
            
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        accumulated_grad_vectors = collect_accumulated_gradients(model, train_loader, device, criterion, accumulation_steps, accumulation_steps*5, optimizer) 
        grad_vectors = collect_gradients(model, train_loader, device, criterion, 5)

        G_big = torch.norm(torch.mean(torch.stack(accumulated_grad_vectors), dim=0))**2
        grad_norms_squared = torch.tensor([torch.norm(G_local)**2 for G_local in grad_vectors])
        G_small = torch.mean(grad_norms_squared)

        G2 = (1 / (B_big - B_small)) * (B_big * G_big - B_small * G_small)
        S =  (B_small * B_big / (B_small - B_big)) * (G_big - G_small)
        B_simple = S / G2
        B_simples.append(B_simple)

        if log:
            log(INFO, f"Epoch {epoch+1}/{centralized_train_cfg['epochs']}, B_simple: {B_simple:.2f}, B_big: {B_big}, B_small: {B_small}")

    return B_simples





def start_seeded_simulation(
    client_fn: Callable[[str], Client],
    num_clients: int,
    config: ServerConfig,
    strategy: Strategy,
    name: str,
    return_all_parameters: bool = False,
    seed: int = Seeds.DEFAULT,
    iteration: int = 0,
    client_manager: Optional[ClientManager] = None,
    server: Optional[ReturnParametersServer] = None,
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
        client_manager=client_manager,
        server=server,
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
    use_target_accuracy: bool = False,
    target_accuracy: float = 0.6,
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

    if use_target_accuracy:
        client_manager = SimpleClientManager()
        server = TargetAccuracyServer(client_manager=client_manager, strategy=strategy, target_accuracy=target_accuracy)
    else:
        server = None
        client_manager = None

    parameters_for_each_round, hist = start_seeded_simulation(
        client_fn=simulator_client_generator,
        num_clients=num_total_clients,
        config=cfg,
        strategy=strategy,
        name="fedavg",
        return_all_parameters=True,
        seed=Seeds.DEFAULT,
        client_manager=client_manager,
        server=server,
    )
    return parameters_for_each_round, hist
