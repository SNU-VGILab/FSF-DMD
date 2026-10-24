import logging
import os
import random
import re
from collections import OrderedDict
from contextlib import contextmanager
from glob import glob
from pathlib import Path

import numpy as np
import safetensors
import torch
import torch.distributed as dist
import torch.nn as nn
import torch_fidelity
from PIL import Image
from tqdm import tqdm


def center_crop(pil_image: Image.Image, image_size: int) -> Image.Image:
    # Copied from openai/guided-diffusion:8fb3ad9197f16bbc40620447b2742e13458d2831
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def set_seed(seed: int):
    # For random number generation in CPU and GPU
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Ensure that CUDA algorithms are deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Control over external libraries (if applicable)
    os.environ["PYTHONHASHSEED"] = str(seed)


@contextmanager
def fixed_seed(seed: int):
    states = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all(),
    )
    set_seed(seed)
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        torch.cuda.set_rng_state_all(states[3])


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def remove_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    new_state_dict = {}
    for k, v in state_dict.items():
        while k.startswith("module."):
            k = k[len("module.") :]
        new_state_dict[k] = v
    return new_state_dict


def create_logger(logging_dir: str, logger_name: str, use_color: bool = True):
    # Ensure distributed environment is initialized
    rank = dist.get_rank() if dist.is_initialized() else 0
    logger = logging.getLogger(logger_name)

    # Prevent adding handlers multiple times
    if not logger.handlers:
        if rank == 0:  # Main process
            logger.setLevel(logging.INFO)

            # Ensure log directory exists
            os.makedirs(logging_dir, exist_ok=True)
            log_file = os.path.join(logging_dir, logger_name + "_log.txt")

            # Create file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)

            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            if use_color:
                console_formatter = ColoredFormatter(
                    "%(levelname)s: %(message)s",
                )
            else:
                console_formatter = logging.Formatter(
                    "%(levelname)s: %(message)s",
                )
            console_handler.setFormatter(console_formatter)

            # Add handlers to logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

            # Prevent log messages from propagating to the root logger
            logger.propagate = False
        else:  # Non-main process
            logger.setLevel(logging.WARNING)  # Set to higher level to avoid output
            logger.addHandler(logging.NullHandler())

    return logger


class ColoredFormatter(logging.Formatter):
    COLOR_MAP = {
        "DEBUG": "\033[37m",  # White
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Purple
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class ImgLatentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        latent_norm: bool = True,
        latent_multiplier: float = 1.0,
    ):
        self.data_dir = data_dir
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier

        self.files = sorted(glob(os.path.join(data_dir, "*.safetensors")))
        self.img_to_file_map = self.get_img_to_safefile_map()

        if latent_norm:
            self._latent_mean, self._latent_std = self.get_latent_stats()
        else:
            self._latent_mean, self._latent_std = torch.tensor(0), torch.tensor(1)

    def get_img_to_safefile_map(self) -> dict[int, dict[str, str | int]]:
        img_to_file = {}
        for safe_file in self.files:
            with safetensors.safe_open(safe_file, framework="pt", device="cpu") as f:
                labels = f.get_slice("labels")
                labels_shape = labels.get_shape()
                num_imgs = labels_shape[0]
                cur_len = len(img_to_file)
                for i in range(num_imgs):
                    img_to_file[cur_len + i] = {
                        "safe_file": safe_file,
                        "idx_in_file": i,
                    }
        return img_to_file

    def get_latent_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        pattern = r"/data/([^/]+)/[^_]+_(\d+)$"
        match = re.search(pattern, self.data_dir)
        vae_name, res = match.group(1), match.group(2)
        latent_stats_cache_file = os.path.join(
            "./buffers/vaes/stat", f"{vae_name}_{res}.pt"
        )

        if not os.path.exists(latent_stats_cache_file):
            latent_stats = self.compute_latent_stats()
            torch.save(latent_stats, latent_stats_cache_file)
        else:
            latent_stats = torch.load(latent_stats_cache_file)
        return latent_stats["mean"], latent_stats["std"]

    def compute_latent_stats(self) -> dict[str, torch.Tensor]:
        num_samples = min(10000, len(self.img_to_file_map))
        random_indices = np.random.choice(
            len(self.img_to_file_map), num_samples, replace=False
        )
        latents = []
        for idx in tqdm(random_indices):
            img_info = self.img_to_file_map[idx]
            safe_file, img_idx = img_info["safe_file"], img_info["idx_in_file"]
            with safetensors.safe_open(safe_file, framework="pt", device="cpu") as f:
                features = f.get_slice("latents")
                feature = features[img_idx : img_idx + 1]
                latents.append(feature)

        latents = torch.cat(latents, dim=0)
        mean = latents.mean(dim=[0, 2, 3], keepdim=True)
        std = latents.std(dim=[0, 2, 3], keepdim=True)
        latent_stats = {"mean": mean, "std": std}
        return latent_stats

    def __len__(self) -> int:
        return len(self.img_to_file_map.keys())

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        img_info = self.img_to_file_map[idx]
        safe_file, img_idx = img_info["safe_file"], img_info["idx_in_file"]
        with safetensors.safe_open(safe_file, framework="pt", device="cpu") as f:
            tensor_key = "latents" if np.random.uniform(0, 1) > 0.5 else "latents_flip"
            features = f.get_slice(tensor_key)
            labels = f.get_slice("labels")
            feature = features[img_idx : img_idx + 1]
            label = labels[img_idx : img_idx + 1]

        if self.latent_norm:
            feature = (feature - self._latent_mean) / self._latent_std
        feature = feature * self.latent_multiplier

        feature = feature.squeeze(0)
        label = label.squeeze(0)
        return feature, label


class FIDDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor):
        self.images = data

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> torch.Tensor:
        img = self.images[idx]
        return (img * 255 + 0.5).clamp(0, 255).to(torch.uint8)


def torch_fidelity_metrics(data: torch.Tensor, fid_reference_file: str):
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=FIDDataset(data),
        input2=None,
        fid_statistics_file=fid_reference_file,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=True,
    )
    return metrics_dict
