import torch
import torchvision.transforms as transforms
from diffusers.models import AutoencoderKL

from utils import center_crop


class SD_VAE:
    def __init__(self, name: str, img_size: int = 256):
        self.img_size = img_size
        self.load(name)

    def load(self, name: str):
        name2type = {
            "sdvae_ema_f8c4": "stabilityai/sd-vae-ft-ema",
            "sdvae_f8c4": "stabilityai/sd-vae-ft-mse",
        }

        if name not in name2type:
            raise KeyError(
                f"Unknown model name: {name}. Available options: {list(name2type.keys())}"
            )

        self.model = AutoencoderKL.from_pretrained(name2type[name])
        self.model.cuda().eval()
        return self

    def img_transform(self, p_hflip: float = 0, img_size: int | None = None):
        img_size = img_size if img_size is not None else self.img_size
        img_transforms = [
            transforms.Lambda(lambda pil_image: center_crop(pil_image, img_size)),
            transforms.RandomHorizontalFlip(p=p_hflip),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
        return transforms.Compose(img_transforms)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            posterior = self.model.encode(images.cuda()).latent_dist.mean
            return posterior

    def decode_to_images(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            images = torch.clip((self.model.decode(z.cuda()).sample + 1) / 2, 0, 1)
        return images
