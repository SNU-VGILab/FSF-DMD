<div align="center">
  <h1>[NeurIPS 2026] Distribution Matching Distillation without Fake Score Network</h1>
  <a href="https://arxiv.org/abs/2605.19256">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2605.19256-b31b1b" />
  </a>
  <a href="https://snu-vgilab.github.io/FSF-DMD/">
    <img alt="Project Page" src="https://img.shields.io/badge/Project%20Page-FSF--DMD-00B894" />
  </a>
  <a href="https://huggingface.co/YoungJoong/FSF-DMD">
    <img alt="Hugging Face" src="https://img.shields.io/badge/Weights-Hugging%20Face-FFD21E" />
  </a>
  <a href="#citation">
    <img alt="BibTeX" src="https://img.shields.io/badge/Citation-BibTeX-8250df" />
  </a>
  <br />
  <br />
  <a href="https://revsic.github.io/">Youngjoong Kim</a><sup>1</sup>,
  Deokyeong Lee<sup>2</sup>,
  <a href="https://jaesik.info/">Jaesik Park</a><sup>1&dagger;</sup>
  <br />
  <sup>1</sup><strong>Seoul National University</strong>,
  <sup>2</sup><strong>Sogang University</strong>
  <br />
  <sup>&dagger;</sup> Corresponding author.
  <br />
  <br />
  <p align="center">
    <img src="sample.png" alt="FSF-DMD sample images" width="90%">
  </p>
</div>

## Requirements

This repository has been tested in the following environment:
- Python 3.12.3, CUDA 13, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition

Python dependencies are managed with [uv](https://docs.astral.sh/uv/) (see [pyproject.toml](./pyproject.toml)).
```bash
uv sync
# (optional) for scripts/convert_imf.py
uv sync --group convert
```

## Preparation

1. Download the VAE checkpoints, latent statistics, and FID reference files from [Hugging Face](https://huggingface.co/YoungJoong/FSF-DMD).
2. Place the downloaded `outputs` and `buffers` directories at the top level of the repository.
3. Preprocess the dataset (skip this step if you are not training networks):
    ```bash
    uv run accelerate launch \
        --num_processes 1 \
        --num_machines 1 \
        --mixed_precision bf16 \
        data.py \
        --data_path ./ILSVRC2012_img_train \
        --config vavae_f16d32 \
        --image_size 256 \
        --batch_size 20 \
        --num_workers 1
    ```
4. Log in to [Weights & Biases](https://wandb.ai) (`wandb login`) for training logs.

To finetune the official iMF-B/2:
1. Download [the official orbax checkpoint](https://huggingface.co/Lyy0725/iMF/blob/main/iMF-B-2-full.zip).
2. Place it at `outputs/iMF-B-2/iMF-B-2-full.zip` and unzip it.
3. Convert it to the PyTorch layout (see [networks/imfdit.py](./networks/imfdit.py)):
    ```bash
    uv run --group convert scripts/convert_imf.py \
        --input outputs/iMF-B-2/iMF-B-2-full \
        --config configs/fsfdmd-imf-official.yaml \
        --output outputs/iMF-B-2/checkpoints/iMF-B-2-full-converted.pt
    ```
4. Preprocess the dataset with `--config sdvae_f8c4`.

## Usage

This repository provides configurations in [configs](./configs):

| Config | Network | Checkpoint | FID50K |
| --- | --- | --- | --- |
| [fsfdmd-distill.yaml](./configs/fsfdmd-distill.yaml) | FSF-DMD-DiT-B/1 (VA-VAE) | [2026.10.22KST20.04.12-fsfdmd-distill](https://huggingface.co/YoungJoong/FSF-DMD/tree/main/outputs/fsfdmd/2026.10.22KST20.04.12-fsfdmd-distill) | 3.8942 |
| [fsfdmd-imf-official.yaml](./configs/fsfdmd-imf-official.yaml) | iMF-DiT-B/2 (SD-VAE) | [2026.10.24KST16.38.59-fsfdmd-imf-official](https://huggingface.co/YoungJoong/FSF-DMD/tree/main/outputs/fsfdmd/2026.10.24KST16.38.59-fsfdmd-imf-official) | 3.2964 |

Train few-step models with FSF-DMD:

```bash
# Use the training script (CONFIG is optional).
bash scripts/train.sh

# Or run the training command directly.
uv run accelerate launch \
    --dynamo_backend=no \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    train.py \
    --config ./configs/fsfdmd-distill.yaml
```

Evaluate models by calculating FID and Inception Score:
```bash
TARGET=2026.10.22KST20.04.12-fsfdmd-distill bash scripts/eval.sh
```

## License

This repository is released under the non-commercial research and educational use license in [LICENSE](LICENSE). It is a source-available research release and is not distributed under an OSI-approved open-source license.

The FSF-DMD research license applies only to the original code in this repository that is owned by the authors. Third-party software, models, weights, and datasets remain under their own license terms. See [THIRD_PARTY.md](THIRD_PARTY.md) for a concise inventory.

## Acknowledgements

This repository is mainly based on [LINs-lab/UCGM](https://github.com/LINs-lab/UCGM), [LTH14/JiT](https://github.com/LTH14/JiT), [Lyy-iiis/imeanflow](https://github.com/Lyy-iiis/imeanflow), and [hustvl/LightningDiT](https://github.com/hustvl/LightningDiT). We thank the authors for their excellent projects.

## Citation

```bibtex
@inproceedings{kim2026distribution,
  title={Distribution Matching Distillation without Fake Score Network},
  author={Youngjoong Kim and Deokyeong Lee and Jaesik Park},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026},
}
```
