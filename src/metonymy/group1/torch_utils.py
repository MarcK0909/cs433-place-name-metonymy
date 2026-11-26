"""Device choice, the GPU rule of the lab server, seeds, memory and size.

Lab server rule (slides): never use GPU 3, it is booked for a service. Every
script that uses PyTorch calls choose_device(), which stops when GPU 3 is
visible or when several GPUs are visible and none was chosen.
"""
import os
import platform
import random
import resource
import sys

# Number GPUs as nvidia-smi does, so CUDA_VISIBLE_DEVICES=1 means "GPU 1" in
# nvidia-smi. Must be set before CUDA starts.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np  # noqa: E402
import torch  # noqa: E402

FORBIDDEN_GPUS = {"3"}


def check_gpu_choice(visible, device_count):
    """Return an error message if this GPU choice breaks the lab rule, else None.

    visible: the value of CUDA_VISIBLE_DEVICES (None when unset).
    """
    if visible is None:
        if device_count > 1:
            return (
                "Several GPUs are visible. Choose one free GPU first, for example "
                "CUDA_VISIBLE_DEVICES=1 (check nvidia-smi; never GPU 3)."
            )
        return None
    chosen = {part.strip() for part in visible.split(",") if part.strip()}
    if chosen & FORBIDDEN_GPUS:
        return "GPU 3 is booked for a service on the lab server. Choose another GPU."
    return None


def choose_device(requested="auto"):
    """cuda, mps or cpu; requested may force one of them."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        problem = check_gpu_choice(os.environ.get("CUDA_VISIBLE_DEVICES"), torch.cuda.device_count())
        if problem:
            sys.exit(problem)
        return torch.device("cuda")
    if requested == "cuda":
        sys.exit("CUDA was requested but no GPU is available.")
    if requested in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "mps":
        sys.exit("MPS was requested but is not available.")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device):
    """Peak GPU memory (cuda), memory held by the MPS driver (mps), or the
    peak memory of the whole process (cpu), in MB."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # bytes on macOS, KB on Linux


def device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"Apple GPU (MPS), {platform.machine()}"
    return f"CPU, {platform.machine()}"


def parameter_counts(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}


def synchronize(device):
    """Wait for the GPU, so that timings are right."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
