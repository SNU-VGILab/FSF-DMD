#!/usr/bin/env python
"""Convert an official iMF (Lyy-iiis/imeanflow) orbax checkpoint to a PyTorch imfDiT checkpoint.

Usage:
    python scripts/convert_imf.py \
        --input outputs/iMF-B-2/iMF-B-2-full \
        --config configs/fsfdmd-imf-official.yaml \
        --output outputs/iMF-B-2/checkpoints/iMF-B-2-full-converted.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from networks import NETWORKS


def source_path_for_torch_key(key):
    parts = key.split(".")
    src = ["net"]
    i = 0
    while i < len(parts):
        part = parts[i]
        if part in {"shared_blocks", "u_heads", "v_heads"}:
            src.append(f"{part}_{parts[i + 1]}")
            i += 2
        elif part == "mlp" and i + 1 < len(parts) and parts[i + 1] in {"0", "2"}:
            src.extend(["mlp", f"layers_{parts[i + 1]}"])
            i += 2
        elif part == "weight":
            prev = parts[i - 1]
            if prev == "_flax_linear":
                src.append("kernel")
            elif prev == "_flax_embedding":
                src.append("embedding")
            elif prev == "proj":
                src.append("kernel")
            else:
                src.append("kernel")
            i += 1
        else:
            src.append(part)
            i += 1
    return src


def get_leaf(tree, path):
    node = tree
    for part in path:
        node = node[part]
    return node


def to_torch_tensor(array, torch_key):
    tensor = torch.from_numpy(np.asarray(array))
    if torch_key.endswith("._flax_linear.weight"):
        tensor = tensor.t().contiguous()
    elif torch_key == "x_embedder.proj.weight":
        tensor = tensor.permute(3, 2, 0, 1).contiguous()
    else:
        tensor = tensor.contiguous()
    return tensor


def convert_tree(tree, reference_state_dict):
    converted = {}
    for key, ref in reference_state_dict.items():
        tensor = to_torch_tensor(get_leaf(tree, source_path_for_torch_key(key)), key)
        if tuple(tensor.shape) != tuple(ref.shape):
            raise ValueError(
                f"Shape mismatch for {key}: got {tuple(tensor.shape)}, expected {tuple(ref.shape)}"
            )
        converted[key] = tensor.to(dtype=ref.dtype)
    return converted


def convert_optimizer_state(orbax_opt_state, reference_model, optimizer_config):
    optimizer = torch.optim.AdamW(
        reference_model.parameters(),
        lr=optimizer_config["lr"],
        weight_decay=optimizer_config["weight_decay"],
        betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
    )
    opt_state = optimizer.state_dict()
    param_ids = opt_state["param_groups"][0]["params"]
    names = [name for name, _ in reference_model.named_parameters()]

    adam_state = orbax_opt_state["0"]
    step = torch.tensor(float(np.asarray(adam_state["count"])), dtype=torch.float32)
    opt_state["state"] = {}
    for param_id, name in zip(param_ids, names, strict=True):
        opt_state["state"][param_id] = {
            "step": step.clone(),
            "exp_avg": to_torch_tensor(
                get_leaf(adam_state["mu"], source_path_for_torch_key(name)), name
            ),
            "exp_avg_sq": to_torch_tensor(
                get_leaf(adam_state["nu"], source_path_for_torch_key(name)), name
            ),
        }
    return opt_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/iMF-B-2/iMF-B-2-full")
    parser.add_argument("--config", default="configs/fsfdmd-imf-official.yaml")
    parser.add_argument(
        "--output",
        default="outputs/iMF-B-2/checkpoints/iMF-B-2-full-converted.pt",
    )
    parser.add_argument("--no-opt", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output)
    config = yaml.safe_load(Path(args.config).read_text())

    model_config = dict(config["model"])
    if "in_chans" in model_config:
        model_config["in_channels"] = model_config.pop("in_chans")
    model = NETWORKS[model_config.pop("type")](
        input_size=config["data"]["image_size"] // config["vae"]["downsample_ratio"],
        num_classes=config["data"]["num_classes"],
        **model_config,
    )
    reference_state = model.state_dict()

    state = ocp.PyTreeCheckpointer().restore(input_path)
    checkpoint = {
        "model": convert_tree(state["params"], reference_state),
        "ema": convert_tree(state["ema_params"], reference_state),
        "config": config,
        "steps": int(np.asarray(state["step"])),
    }
    if not args.no_opt:
        checkpoint["opt"] = convert_optimizer_state(
            state["opt_state"], model, config["optimizer"]
        )

    model.load_state_dict(checkpoint["model"])
    model.load_state_dict(checkpoint["ema"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"saved {output_path}")
    print(f"steps {checkpoint['steps']}")
    print(f"model keys {len(checkpoint['model'])}")
    print(f"ema keys {len(checkpoint['ema'])}")
    print(f"has opt {'opt' in checkpoint}")


if __name__ == "__main__":
    main()
