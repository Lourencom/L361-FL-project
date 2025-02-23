import torch
from logging import INFO, DEBUG
from flwr.common import log, NDArrays, Scalar, Parameters, ndarrays_to_parameters
import numpy as np


def compute_critical_batch(noise_scales: list, constant: float = 1.0) -> float:
    # simple avg of noise scales
    avg_noise_scale = np.mean(noise_scales)
    eps = 1e-8
    
    # Computing an estimated critical batch size (Bcrit) using a simple heuristic.
    critical_batch_size = constant / (avg_noise_scale + eps)
    return critical_batch_size


def compute_noise_scale_from_gradients(grad_list, eps=1e-6):
    """
    Compute the noise scale (Bsimple) from a list of gradient vectors.
    
    Parameters:
        grad_list (list[Tensor]): List of gradient vectors.
        eps (float): Small constant for numerical stability.
    
    Returns:
        float: Estimated noise scale.
    """
    try:
        if not grad_list:
            return None

        # Stack gradients: shape (num_batches, num_params)
        grad_stack = torch.stack(grad_list)
        mean_grad = grad_stack.mean(dim=0)
        # Compute average variance per parameter element.
        var_grad = grad_stack.var(dim=0, unbiased=False).mean()
        denom = mean_grad.norm()**2 + eps
        noise_scale = var_grad / denom
        return noise_scale.item()
    except Exception as e:
        log(DEBUG, "Error in compute_noise_scale_from_gradients: %s", e)
        return None


def get_gradient_vector(model, data, target, loss_fn, device):
    model.zero_grad()
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    if grads:
        return torch.cat(grads)
    log(DEBUG, "No gradients found")
    return None


def collect_gradients(model, train_loader, device, criterion, num_mini_batches):
    grad_vectors = []
    for i, (data, target) in enumerate(train_loader):
        if i >= num_mini_batches:
            break
        grad_vector = get_gradient_vector(model, data, target, criterion, device)
        if grad_vector is not None:
            grad_vectors.append(grad_vector)
    return grad_vectors
