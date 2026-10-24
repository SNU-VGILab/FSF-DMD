from dataclasses import dataclass
from typing import Protocol, Self, runtime_checkable

import numpy as np
import torch
import torch.nn as nn

from networks.dit import AugmentedDiT


@runtime_checkable
class FlowMap(Protocol):
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        *args,
        y: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor: ...


@runtime_checkable
class Trainable(Protocol):
    def loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        c: torch.Tensor,
        train_steps: int,
        ema: nn.Module | None = None,
        teacher: nn.Module | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]: ...

    def sample(
        self,
        z: torch.Tensor,
        model: nn.Module,
        c: torch.Tensor,
        sampling_steps: int = 2,
        precfg: torch.Tensor | None = None,
        cfg_start: torch.Tensor | None = None,
        cfg_end: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


@dataclass
class Context:
    omega: torch.Tensor
    t_min: torch.Tensor | float
    t_max: torch.Tensor | float

    t: torch.Tensor
    masked: torch.Tensor | None = None

    def __post_init__(self):
        if self.masked is None:
            self.masked = self.omega.masked_fill(
                torch.logical_or(self.t < self.t_min, self.t >= self.t_max), 1.0
            )

    @property
    def kwargs(self) -> dict[str, torch.Tensor]:
        return {}

    def new_t(self, t: torch.Tensor) -> Self:
        return Context(self.omega, self.t_min, self.t_max, t)


@runtime_checkable
class ContextManagable(Protocol):
    explicit_cfg: bool
    BaseContext: type[Context]

    def __init__(
        self, model: nn.Module, precfg: float, cfgrange: tuple[float, float]
    ): ...

    def v_t(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        ctx: Context | None = None,
        **kwargs,
    ) -> torch.Tensor: ...

    def sample(self, t: torch.Tensor) -> Context: ...

    def fixed(self, t: torch.Tensor) -> Context: ...

    def cfg(
        self, u: torch.Tensor, v: torch.Tensor, s: torch.Tensor | Context
    ) -> torch.Tensor: ...


class AugmentedDiTSupport(ContextManagable):
    class BaseContext(Context):
        @property
        def kwargs(self) -> dict[str, torch.Tensor]:
            return dict(omega=self.masked, t_min=self.t_min, t_max=self.t_max)

        def new_t(self, t: torch.Tensor) -> Self:
            return AugmentedDiTSupport.BaseContext(
                self.omega, self.t_min, self.t_max, t
            )

    def __init__(self, model: nn.Module, precfg: float, cfgrange: tuple[float, float]):
        self.model = model
        self.explicit_cfg = False
        self.precfg = precfg
        self.t_min, self.t_max = cfgrange

    def v_t(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        ctx: Context | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if ctx is not None:
            kwargs.update(ctx.kwargs)
        return self.model(x_t, t, t, **kwargs, y=y)

    def sample(self, t: torch.Tensor) -> Context:
        omega = (torch.rand_like(t) * np.log(8.0)).exp()
        t_min = torch.rand_like(omega) * 0.5
        t_max = torch.rand_like(omega) * 0.5 + 0.5
        return AugmentedDiTSupport.BaseContext(omega, t_min, t_max, t)

    def fixed(self, t: torch.Tensor) -> Context:
        omega = torch.full_like(t, self.precfg)
        t_min = torch.full_like(t, self.t_min)
        t_max = torch.full_like(t, self.t_max)
        return AugmentedDiTSupport.BaseContext(omega, t_min, t_max, t)

    def cfg(
        self, u: torch.Tensor, v: torch.Tensor, s: torch.Tensor | Context
    ) -> torch.Tensor:
        if isinstance(s, Context):
            s = s.masked
        return u + s[:, None, None, None] * (v - u)


def context_manager(
    network: nn.Module,
    precfg: float = 1.0,
    cfgrange: tuple[float, float] = (0.0, 1.0),
) -> ContextManagable:
    while hasattr(network, "module"):
        network = network.module  # destruct wrappers (e.g., DDP)
    match network:
        case AugmentedDiT():
            return AugmentedDiTSupport(network, precfg, cfgrange)
        case _:
            raise NotImplementedError(
                f"a context manager for the type {type(network)} is not implemented"
            )


def default_sample(
    z: torch.Tensor,
    model: nn.Module,
    c: torch.Tensor,
    sampling_steps: int,
    precfg: torch.Tensor | float,
    cfg_start: torch.Tensor | float | None = None,
    cfg_end: torch.Tensor | float | None = None,
    cfgrange: tuple[float, float] | None = None,
) -> torch.Tensor:
    device = z.device
    bsize, *_ = z.shape
    # assign default values
    if isinstance(precfg, float):
        precfg = torch.tensor([precfg] * bsize).to(z)

    if cfg_start is None:
        assert cfgrange is not None
        cfg_start, _ = cfgrange
    if isinstance(cfg_start, float):
        cfg_start = torch.tensor([cfg_start] * bsize).to(z)

    if cfg_end is None:
        assert cfgrange is not None
        _, cfg_end = cfgrange
    if isinstance(cfg_end, float):
        cfg_end = torch.tensor([cfg_end] * bsize).to(z)

    # uniform sampling
    t_steps = torch.linspace(1.0, 0.0, sampling_steps + 1, device=device)
    manager = context_manager(model)
    ctx = manager.BaseContext(precfg, cfg_start, cfg_end, t_steps[0].repeat(bsize))

    x_cur = z
    samples = [x_cur.clone()]
    # assert isinstance(model, FlowMap)
    for t_cur, t_next in zip(t_steps, t_steps[1:]):
        bt, bs = t_cur.repeat(bsize), t_next.repeat(bsize)
        ctx = ctx.new_t(bt)
        F_th_t_s = model(x_cur, bt, bs, **ctx.kwargs, y=c)
        x_cur = x_cur + (t_next - t_cur) * F_th_t_s
        samples.append(x_cur)
    return torch.stack(samples, dim=0)
