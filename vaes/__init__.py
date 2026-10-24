from vaes.identity import ID_AE
from vaes.sdvae import SD_VAE
from vaes.vavae import VA_VAE


VAES = {
    "idae_f1c3": ID_AE,
    "sdvae_f8c4": SD_VAE,
    "sdvae_ema_f8c4": SD_VAE,
    "vavae_f16d32": VA_VAE,
}
