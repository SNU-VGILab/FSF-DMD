import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torch_fidelity
from accelerate import Accelerator
from torchvision.datasets import ImageFolder
from torch_fidelity.utils import (
    create_feature_extractor,
    extract_featuresdict_from_input_id_cached,
    resolve_feature_layer_for_metric,
)
from tqdm import tqdm

from methods import METHODS
from networks import NETWORKS
from utils import FIDDataset, center_crop, set_seed
from vaes import VAES

accelerator = Accelerator()

_PREDEFINED_TARGETS = []
_PREDEFINED_CHECKPOINTS = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mother", type=Path, default=Path("./outputs/fsfdmd"))
    parser.add_argument("--targets", nargs="*", default=_PREDEFINED_TARGETS)
    parser.add_argument("--checkpoints", nargs="*", default=_PREDEFINED_CHECKPOINTS)

    parser.add_argument("--sampling-steps", type=int, default=None)

    parser.add_argument("--precfg", type=float, default=None)
    parser.add_argument("--cfg-start", type=float, default=None)
    parser.add_argument("--cfg-end", type=float, default=None)

    parser.add_argument("--dump", type=Path, default=Path("fid50k.json"))
    parser.add_argument("--total-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=None)

    parser.add_argument("--fid-reference-file", default="./buffers/refs/in1k256_fid_ref.npz")  # fmt: skip

    parser.add_argument("--recall", action="store_true", default=False)
    parser.add_argument("--nhood-size", type=int, default=3)
    parser.add_argument("--prc-batch-size", type=int, default=5000)
    parser.add_argument("--prc-subbatch-size", type=int, default=10000)
    parser.add_argument("--max-real", type=int, default=200000)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--feature-layer-prc", default=None)
    parser.add_argument("--cache-root", type=Path, default=Path("~/.torch_fidelity_cache"))  # fmt: skip
    parser.add_argument("--dataset", type=Path, default=Path("./ILSVRC2012_img_train"))  # fmt: skip
    parser.add_argument("--dataset-cache", default="ILSVRC2012_img_train")
    return parser.parse_args()


def main():
    args = parse_args()
    targets = [args.mother / target for target in args.targets]
    assert len(targets) == len(set(targets))

    # representative
    r, *_ = targets
    assert r.exists()

    c = json.loads((r / "config.json").read_text())
    image_size = c["data"]["image_size"]
    num_classes = c["data"]["num_classes"]
    latent_norm = c["data"]["latent_norm"]
    latent_multiplier = c["data"]["latent_multiplier"]
    method_config = {**c["method"]}
    model_type = c["model"]["type"]
    in_channels = c["model"]["in_channels"]
    vae_type = c["vae"]["type"]
    downsample_ratio = c["vae"]["downsample_ratio"]
    sampling_steps = args.sampling_steps or c["sample"]["sampling_steps"]
    # match train.py: the noise drawn per batch (and thus FID) depends on the batch size
    if args.batch_size is None:
        args.batch_size = c["sample"]["per_batch_size"]
    assert args.batch_size < args.total_samples

    cfg_range = None
    if args.cfg_start is not None and args.cfg_end is not None:
        cfg_range = [args.cfg_start, args.cfg_end]

    accelerator.print(f"Total: {len(targets)}")
    accelerator.print(f"[*DEBUG] PRECFG={args.precfg}, CFG_RANGE={cfg_range}")
    accelerator.print(f"[*DEBUG] Dump to {args.dump}")
    accelerator.print(f"[*DEBUG] Dataset cache: {args.dataset_cache}")

    latent_size = image_size // downsample_ratio
    args.image_size = image_size  # for precision/recall

    if args.precfg is not None:
        method_config["precfg"] = args.precfg
    if cfg_range is not None:
        method_config["cfgrange"] = cfg_range
    method_type = method_config.pop("type")
    method = METHODS[method_type](**method_config)
    model = NETWORKS[model_type](
        input_size=latent_size,
        num_classes=num_classes,
        in_channels=in_channels,
    )
    vae = VAES[vae_type](vae_type)

    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)

    device = accelerator.device
    n_proc = accelerator.num_processes
    accelerator.print(f"[*DEBUG] Total process: {n_proc}")

    n_samples_per_proc = -(-args.total_samples // n_proc)
    n_iters = -(-n_samples_per_proc // args.batch_size)
    offset = n_samples_per_proc * accelerator.process_index

    labels = torch.arange(num_classes)
    labels = labels.repeat(
        -(-(args.total_samples + args.batch_size + n_proc) // len(labels))
    )

    if latent_norm:
        mean_std = torch.load(
            f"./buffers/vaes/stat/{vae_type}_{image_size}.pt",
            weights_only=True,
        )
        latent_mean, latent_std = mean_std["mean"], mean_std["std"]
    else:
        latent_mean, latent_std = torch.tensor(0), torch.tensor(1)

    latent_mean = latent_mean.clone().detach().to(device)
    latent_std = latent_std.clone().detach().to(device)

    for mother in targets:
        mother = Path(mother)
        assert mother.exists()

        config = json.loads((mother / "config.json").read_text())
        assert config["data"]["image_size"] == image_size, mother
        assert config["data"]["num_classes"] == num_classes, mother
        assert config["data"]["latent_norm"] == latent_norm, mother
        assert config["data"]["latent_multiplier"] == latent_multiplier, mother
        assert config["model"]["type"] == model_type, mother
        assert config["model"]["in_channels"] == in_channels, mother
        assert config["vae"]["type"] == vae_type, mother
        assert config["vae"]["downsample_ratio"] == downsample_ratio, mother

        for ckpt in args.checkpoints:
            if ckpt == "top":
                records = [p.name for p in (mother / "checkpoints").glob("top.*.pt")]
                if not records:
                    accelerator.print(
                        f"[*DEBUG] WARNING: a top record not found for {mother.name}"
                    )
                    continue
                ckpt = min(records, key=lambda r: float(r[4:-3]))
            if not (mother / "checkpoints" / ckpt).exists():
                accelerator.print(
                    f"[*DEBUG] WARNING: {ckpt} not found for {mother.name}"
                )

    metrics = {}
    if args.dump.exists():
        metrics.update(json.loads(args.dump.read_text()))

    for mother in (pbar := tqdm(targets, disable=not accelerator.is_main_process)):
        mother = Path(mother)
        config = json.loads((mother / "config.json").read_text())

        pbar.set_postfix_str(mother.name)
        if mother.name not in metrics:
            metrics[mother.name] = {}

        for ckpt in tqdm(
            args.checkpoints, leave=False, disable=not accelerator.is_main_process
        ):
            if ckpt == "top":
                records = [p.name for p in (mother / "checkpoints").glob("top.*.pt")]
                ckpt = min(records, key=lambda r: float(r[4:-3]))
            if not (mother / "checkpoints" / ckpt).exists():
                continue

            metric_key = ckpt
            if args.precfg is not None or cfg_range is not None:
                metric_key = f"{ckpt}:{args.precfg}:{cfg_range}"
            if metric_key in metrics[mother.name]:
                continue

            checkpoint = torch.load(
                mother / "checkpoints" / ckpt,
                map_location=lambda storage, loc: storage,
                weights_only=True,
            )
            if "ema" in checkpoint:
                checkpoint = checkpoint["ema"]
            model.load_state_dict(checkpoint)
            model.eval()
            model.to(device)

            wrapped = accelerator.prepare(model)

            all_samples = []
            # fixed noise for FID, identical to train.py
            set_seed(config["train"]["global_seed"] + accelerator.process_index)
            for i in tqdm(
                range(n_iters), leave=False, disable=not accelerator.is_main_process
            ):
                _offset = offset + i * args.batch_size
                z = torch.randn(
                    args.batch_size,
                    in_channels,
                    latent_size,
                    latent_size,
                    device=device,
                )
                y = labels[_offset : _offset + args.batch_size].to(device)

                x_cur = method.sample(z, wrapped, y, sampling_steps)[-1]
                samples = (x_cur * latent_std) / latent_multiplier + latent_mean
                samples = vae.decode_to_images(samples)
                samples = accelerator.gather(samples).cpu()
                if accelerator.is_main_process:
                    all_samples.append(samples)

            if accelerator.is_main_process:
                all_samples = torch.cat(all_samples)[: args.total_samples]
                metrics_dict = torch_fidelity.calculate_metrics(
                    input1=FIDDataset(all_samples),
                    input2=None,
                    fid_statistics_file=args.fid_reference_file,
                    cuda=True,
                    isc=True,
                    fid=True,
                    kid=False,
                    prc=False,
                    verbose=True,
                )
                metrics[mother.name][metric_key] = metrics_dict
                with open(args.dump, "w") as f:
                    json.dump(metrics, f, indent=2)

                if args.recall:
                    real_dataset = ImageDataset(args.dataset, args.image_size)
                    metrics_dict = precision_recall(
                        all_samples, real_dataset, args, device
                    )
                    metrics[mother.name][metric_key].update(metrics_dict)
                    with open(args.dump, "w") as f:
                        json.dump(metrics, f, indent=2)

            accelerator.wait_for_everyone()


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root: str | Path, image_size: int):
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: center_crop(img, image_size)),
                transforms.PILToTensor(),
            ]
        )
        self.dataset = ImageFolder(root, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image


def compute_kth_radius(
    manifold_features: torch.Tensor,
    nhood_size: int,
    row_batch_size: int,
    col_batch_size: int,
    desc: str,
):
    kth = []
    N, *_ = manifold_features.shape
    for manifold_batch in tqdm(
        manifold_features.split(row_batch_size),
        desc=desc,
        total=-(-N // row_batch_size),
    ):
        dist_batches = [
            torch.cdist(manifold_batch[None], col_batch[None])[0].cpu()
            for col_batch in manifold_features.split(col_batch_size)
        ]
        dist = torch.cat(dist_batches, dim=1)[:, :N]
        kth.append(
            dist.to(torch.float32).kthvalue(nhood_size + 1).values.to(torch.float16)
        )
    return torch.cat(kth)


def precision_recall(
    samples: torch.Tensor,
    real_input: ImageDataset,
    args: argparse.Namespace,
    device: torch.device,
):
    cache_root = args.cache_root.expanduser()
    kwargs = dict(
        input1=real_input,
        input2=FIDDataset(samples),
        cuda=torch.cuda.is_available(),
        prc=True,
        verbose=True,
        cache=True,
        input1_cache_name=args.dataset_cache,
        batch_size=args.feature_batch_size,
        cache_root=cache_root.as_posix(),
        feature_extractor="vgg16",
        feature_layer_prc=args.feature_layer_prc,
        samples_resize_and_crop=args.image_size,
    )
    feat_layer_name = resolve_feature_layer_for_metric("prc", **kwargs)
    feat_extractor = create_feature_extractor("vgg16", [feat_layer_name], **kwargs)
    real_cache_file = (
        cache_root / f"{args.dataset_cache}-vgg16-features-{feat_layer_name}.pt"
    )
    real_kth_cache_file = cache_root / (
        f"{args.dataset_cache}-vgg16-prc-real-kth-{feat_layer_name}"
        f"-maxreal{args.max_real}-nhood{args.nhood_size}"
        f"-row{args.prc_batch_size}-col{args.prc_subbatch_size}.pt"
    )

    print(f"[*DEBUG] Loading/extracting real features: {real_cache_file} (exists={real_cache_file.exists()})", flush=True)  # fmt: skip
    real_features = extract_featuresdict_from_input_id_cached(1, feat_extractor, **kwargs)[feat_layer_name]  # fmt: skip
    print(f"[*DEBUG] Extracting generated features", flush=True)
    gen_features = extract_featuresdict_from_input_id_cached(2, feat_extractor, **kwargs)[feat_layer_name]  # fmt: skip
    del feat_extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    N_real, *_ = real_features.shape
    N_gen, *_ = gen_features.shape

    if args.max_real is not None and args.max_real < N_real:
        indices = torch.linspace(0, N_real - 1, steps=args.max_real, dtype=torch.long)  # fmt: skip
        real_features = real_features[indices]
        N_real, *_ = real_features.shape

    real_features = real_features.to(torch.float16).to(device)
    gen_features = gen_features.to(torch.float16).to(device)

    print(
        "[*DEBUG] PRC mode: precision+recall, "
        f"real={tuple(real_features.shape)}, gen={tuple(gen_features.shape)}, "
        f"max_real={args.max_real}, "
        f"nhood={args.nhood_size}, row_batch={args.prc_batch_size}, col_batch={args.prc_subbatch_size}",
        flush=True,
    )

    gen_kth = compute_kth_radius(
        manifold_features=gen_features,
        nhood_size=args.nhood_size,
        row_batch_size=args.prc_batch_size,
        col_batch_size=args.prc_subbatch_size,
        desc="gen kNN radius",
    )
    if real_kth_cache_file.exists():
        real_kth = torch.load(
            real_kth_cache_file, map_location="cpu", weights_only=True
        )
    else:
        real_kth = compute_kth_radius(
            manifold_features=real_features,
            nhood_size=args.nhood_size,
            row_batch_size=args.prc_batch_size,
            col_batch_size=args.prc_subbatch_size,
            desc="real kNN radius",
        )
        real_kth_cache_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"[*DEBUG] Caching real kNN radius: {real_kth_cache_file}", flush=True)
        torch.save(real_kth.cpu(), real_kth_cache_file)

    recall_inside = 0
    recall_total = 0
    for probes_batch in tqdm(
        real_features.split(args.prc_batch_size),
        desc="recall: real probes",
        total=-(-N_real // args.prc_batch_size),
    ):
        dist = torch.cat([
            torch.cdist(probes_batch[None], col_batch[None])[0].cpu()
            for col_batch in gen_features.split(args.prc_subbatch_size)
        ], dim=1)  # fmt: skip
        inside = (dist <= gen_kth).any(dim=1)
        recall_inside += int(inside.sum().item())
        recall_total += inside.numel()

    precision_inside = 0
    precision_total = 0
    for probes_batch in tqdm(
        gen_features.split(args.prc_batch_size),
        desc="precision: gen probes",
        total=-(-N_gen // args.prc_batch_size),
    ):
        dist = torch.cat([
            torch.cdist(probes_batch[None], col[None])[0].cpu()
            for col in real_features.split(args.prc_subbatch_size)
        ], dim=1)  # fmt: skip
        inside = (dist <= real_kth).any(dim=1)
        precision_inside += int(inside.sum().item())
        precision_total += inside.numel()

    recall = recall_inside / recall_total
    precision = precision_inside / precision_total
    f_score = 2 * precision * recall / max(1e-5, precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "max_real": N_real,
    }


if __name__ == "__main__":
    main()
