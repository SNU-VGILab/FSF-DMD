import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.protocols import context_manager, default_sample


class FakeScoreFreeDMD:
    def __init__(
        self,
        label_drop: float = 0.1,
        precfg: float = 6.0,
        cfgrange: tuple[float, float] = (0.0, 0.9),
        cd_mask_s: float | None = 0.5,
        cd_use_cond: bool = True,
        unroll_step: int = 2,
        unroll_fullprop: bool = False,
        unroll_memtune: bool = True,
        dmd_indep_z: bool = True,
        dmd_huber: float | None = None,
        dmd_weight: float = 0.01,
        dmd_ada_weight: bool = False,
        dmd_cos_weight: bool = True,
        gamma: float | None = 10.0,
    ):
        self.label_drop = label_drop
        self.precfg = precfg
        self.cfgrange = cfgrange
        self.cd_mask_s = cd_mask_s
        self.cd_use_cond = cd_use_cond
        self.unroll_step = unroll_step
        self.unroll_fullprop = unroll_fullprop
        self.unroll_memtune = unroll_memtune
        self.dmd_indep_z = dmd_indep_z
        self.dmd_huber = dmd_huber
        self.dmd_weight = dmd_weight
        self.dmd_ada_weight = dmd_ada_weight
        self.dmd_cos_weight = dmd_cos_weight
        self.gamma = gamma

    def _destruct_module(self, model: nn.Module):
        _module = model
        while hasattr(_module, "module"):
            _module = _module.module
        return _module

    def loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        c: torch.Tensor,
        train_steps: int,
        ema: nn.Module | None = None,
        teacher: nn.Module | None = None,
    ):
        # sampling timesteps
        bsize, _, _, _ = x.shape
        dist = torch.distributions.Beta(0.8, 1.0)
        t = dist.sample((bsize, 1, 1, 1)).to(x).clamp(0.0, 1.0)
        s = dist.sample((bsize, 1, 1, 1)).to(x).clamp(0.0, 1.0)
        t, s = torch.maximum(t, s), torch.minimum(t, s)

        z = torch.randn_like(x)

        nullc = self._destruct_module(model).num_classes
        mng = context_manager(model, self.precfg, self.cfgrange)
        ctx = mng.sample(bt := t.flatten())

        x_t = (1 - t) * x + t * z

        c_drop = c.clone()
        guide = mng
        if teacher is not None:
            guide = context_manager(teacher)
        else:
            # class-label dropout
            ndrop = round(self.label_drop * len(c))
            c_drop[-ndrop:] = nullc
        if self.cd_mask_s:
            s = torch.where(torch.rand_like(s) < 1 - self.cd_mask_s, s, t)

        bs = s.flatten()
        with torch.no_grad():
            uncond = guide.v_t(x_t, bt, torch.full_like(c, nullc), ctx)
            v_t = guide.v_t(x_t, bt, c_drop, ctx)
            if guide.explicit_cfg:
                v_t = guide.cfg(uncond, v_t, ctx)

            target = v_t
            if self.cd_use_cond:
                target = guide.cfg(uncond, z - x, ctx)

        F_th = model(x_t, bt, bs, **ctx.kwargs, y=c_drop)
        with torch.no_grad():
            eps = 0.005
            dF_dt = (
                model(x_t + eps * v_t, bt + eps, bs, **ctx.kwargs, y=c_drop)
                - model(x_t - eps * v_t, bt - eps, bs, **ctx.kwargs, y=c_drop)
            ) / (2 * eps)

        v_pred = F_th + (t - s) * dF_dt
        cossim = (1.0 - F.cosine_similarity(v_pred, target, dim=1)).mean(dim=(1, 2))
        flow_map_distill_loss = (v_pred - target).square().mean(dim=(1, 2, 3))  # fmt: skip
        flow_map_distill_loss = cossim + (flow_map_distill_loss + 25).sqrt() - 5
        imf_loss = (bt * 1.57).cos() * flow_map_distill_loss

        if self.dmd_weight == 0.0:
            # early exit
            return imf_loss.mean(), {
                "distill": flow_map_distill_loss.mean().item(),
            }

        # DMD
        hat_x = z
        ctx = mng.fixed(bt)
        ts = torch.linspace(1, 0, self.unroll_step + 1).to(x)
        # sampling
        if self.unroll_memtune:
            if self.unroll_step == 1:
                bt_cur = torch.ones(bsize, device=x.device)
                bt_next = torch.zeros(bsize, device=x.device)
                F_th = model(hat_x, bt_cur, bt_next, **ctx.new_t(bt_cur).kwargs, y=c)
                hat_x = z - F_th.detach()
            elif self.unroll_step == 2:
                assert not self.unroll_fullprop
                left, right = z.chunk(2, dim=0)
                ctx = mng.fixed(bt[: bsize // 2])
                # sampling
                bt0, bt1, bt2 = ts[:, None].repeat(1, bsize // 2)
                left01 = model(left, bt0, bt1, **ctx.new_t(bt0).kwargs, y=c[: bsize // 2])  # fmt: skip
                with torch.no_grad():
                    right01 = model(right, bt0, bt1, **ctx.new_t(bt0).kwargs, y=c[bsize // 2 :])  # fmt: skip

                    left = left + (ts[1] - ts[0]) * left01
                    left12 = model(left, bt1, bt2, **ctx.new_t(bt1).kwargs, y=c[: bsize // 2])  # fmt: skip

                    right = right + (ts[1] - ts[0]) * right01

                right12 = model(right, bt1, bt2, **ctx.new_t(bt1).kwargs, y=c[bsize // 2 :])  # fmt: skip

                F_th = torch.cat([
                    (ts[0] - ts[1]) * left01 + (ts[1] - ts[2]) * left12,
                    (ts[0] - ts[1]) * right01 + (ts[1] - ts[2]) * right12,
                ], dim=0)  # fmt: skip

                with torch.no_grad():
                    hat_x = z - F_th

                # rollback
                ctx = mng.fixed(bt)
            else:
                raise RuntimeError(
                    "`unroll_memtune` supports only single or two-step generation"
                )
        else:
            F_th = 0
            indices = torch.arange(bsize, device=x.device)[:, None, None, None]
            for i, (t_cur, t_next) in enumerate(zip(ts, ts[1:])):
                bt_cur, bt_next = t_cur.repeat(bsize), t_next.repeat(bsize)
                pred = model(hat_x, bt_cur, bt_next, **ctx.new_t(bt_cur).kwargs, y=c)
                hat_x = hat_x + (t_next - t_cur) * pred.detach()
                # drop gradient partially
                if not self.unroll_fullprop:
                    pred = torch.where(
                        torch.logical_and(
                            i * bsize // self.unroll_step <= indices,
                            indices < (i + 1) * bsize // self.unroll_step,
                        ),
                        pred,
                        pred.detach(),
                    )
                F_th = F_th + (t_cur - t_next) * pred

        # time-shifted t
        if self.gamma:
            t = torch.rand_like(t)
            t = self.gamma * t / ((self.gamma - 1) * t + 1)
            # to inform the omega
            t = t.clamp_max(ctx.t_max[:, None, None, None] - 1e-3)
            bt = t.flatten()

        # guiding network
        if ema is None:
            ema = model
        if teacher is None:
            guide = context_manager(ema)

        ctx = ctx.new_t(bt)
        if self.dmd_indep_z:
            z = torch.randn_like(z)
        hat_x_t = (1 - t) * hat_x + t * z
        with torch.no_grad():
            teacher = guide.v_t(hat_x_t, bt, c, ctx)
            if guide.explicit_cfg:
                u = guide.v_t(hat_x_t, bt, torch.full_like(c, nullc), ctx)
                teacher = guide.cfg(u, teacher, ctx)

            fake = ema(hat_x_t, bt, torch.zeros_like(bt), **ctx.kwargs, y=c)
            grad = fake - teacher
            if self.dmd_ada_weight:
                grad = grad / (teacher - F_th).abs().mean(dim=(1, 2, 3), keepdim=True)
                grad = torch.nan_to_num(grad)

            dmd_target = F_th - grad

        dmd_loss = (F_th - dmd_target).square().mean(dim=(1, 2, 3))
        if self.dmd_huber:
            dmd_loss = (dmd_loss + self.dmd_huber**2).sqrt() - self.dmd_huber

        w = self.dmd_weight
        if self.dmd_cos_weight:
            w = w * (bt * 1.57).cos()

        # log purpose
        with torch.no_grad():
            logs = {
                "distill": flow_map_distill_loss.mean().item(),
                "dmd": dmd_loss.mean().item(),
            }

        loss = imf_loss + dmd_loss * w
        return loss.mean(), logs

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        model: nn.Module,
        c: torch.Tensor,
        sampling_steps: int = 2,
        precfg: torch.Tensor | None = None,
        cfg_start: torch.Tensor | None = None,
        cfg_end: torch.Tensor | None = None,
    ):
        if precfg is None:
            precfg = self.precfg
        return default_sample(
            z, model, c, sampling_steps, precfg, cfg_start, cfg_end, self.cfgrange
        )


class FakeScoreFreeDMDForiMFOfficial:
    def __init__(
        self,
        label_drop: float = 0.1,
        precfg: float = 7.5,
        cfgrange: tuple[float, float] = (0.4, 0.65),
        unroll_step: int = 1,
        unroll_fullprop: bool = False,
        dmd_indep_z: bool = False,
        dmd_huber: float | None = None,
        dmd_weight: float = 0.01,
        dmd_ada_weight: bool = False,
        dmd_cos_weight: bool = True,
        gamma: float | None = 10.0,
    ):
        super().__init__()
        self.label_drop = label_drop
        self.precfg = precfg
        self.cfgrange = cfgrange
        self.unroll_step = unroll_step
        self.unroll_fullprop = unroll_fullprop
        self.dmd_indep_z = dmd_indep_z
        self.dmd_huber = dmd_huber
        self.dmd_weight = dmd_weight
        self.dmd_ada_weight = dmd_ada_weight
        self.dmd_cos_weight = dmd_cos_weight
        self.gamma = gamma

        self.mean = [0.86488, -0.27787343, 0.21616915, 0.3738409]
        self.std = [4.85503674, 5.31922414, 3.93725398, 3.9870003]

    def _destruct_module(self, model: nn.Module):
        _module = model
        while hasattr(_module, "module"):
            _module = _module.module
        return _module

    def loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        c: torch.Tensor,
        train_steps: int,
        ema: nn.Module | None = None,
        teacher: nn.Module | None = None,
    ):
        m = torch.tensor(self.mean)[:, None, None].to(x)
        s = torch.tensor(self.std)[:, None, None].to(x)
        x = (x - m) / s

        # sampling timesteps
        bsize, _, _, _ = x.shape
        t = (torch.randn(bsize, 1, 1, 1).to(x) * 1.0 - 0.4).sigmoid()
        s = (torch.randn(bsize, 1, 1, 1).to(x) * 1.0 - 0.4).sigmoid()
        t, s = torch.maximum(t, s), torch.minimum(t, s)
        mask = torch.arange(bsize, device=x.device) < int(bsize * 0.5)
        s[mask] = t[mask]

        z = torch.randn_like(x)

        nullc = (raw := self._destruct_module(model)).num_classes

        omega = (torch.rand_like(bt := t.flatten()) * np.log(7.0 + 1.0)).exp()
        t_min = (torch.rand_like(bt) * 0.5).masked_fill(mask, 0.0)
        t_max = (torch.rand_like(bt) * 0.5 + 0.5).masked_fill(mask, 1.0)
        masked = omega.masked_fill(torch.logical_or(bt < t_min, bt > t_max), 1.0)

        x_t = (1 - t) * x + t * z
        v_t = z - x
        guide = model if teacher is None else teacher
        with torch.no_grad():
            zeros = torch.zeros_like(bt)
            ones = torch.ones_like(bt)
            # CFG: 1.0[0.0, 1.0]
            _, v_u = guide(x_t, bt, zeros, ones, zeros, ones, torch.full_like(c, nullc))
            # CFG: omega[0.0, 1.0]
            _, v_c = guide(x_t, bt, zeros, omega, zeros, ones, c)
            v_g_fm = v_t + (1 - 1 / omega[:, None, None, None]) * (v_c - v_u)
            # CFG: masked[0.0, 1.0]
            _, v_c = guide(x_t, bt, zeros, masked, zeros, ones, c)
            v_g = v_t + (1 - 1 / masked[:, None, None, None]) * (v_c - v_u)
            v_g[mask] = v_g_fm[mask]

        # class-label dropout
        ndrop = (torch.rand(bsize, device=x.device) < self.label_drop).sum()
        c_drop = c.clone()
        c_drop[:ndrop] = nullc
        v_g[:ndrop] = v_t[:ndrop]

        u, dudt, v = torch.func.jvp(
            lambda x_t, t, s: raw(x_t, t, t - s, omega, t_min, t_max, y=c_drop),
            (x_t, bt, s.flatten()),
            (v_c, ones, zeros),
            has_aux=True,
        )

        V = u + (t - s) * dudt.detach()
        loss_u = (V - v_g).square().sum(dim=(1, 2, 3))
        loss_v = (v - v_g).square().sum(dim=(1, 2, 3))
        imf_loss = loss_u / (loss_u + 0.01).detach() + loss_v / (loss_v + 0.01).detach()

        # log purpose
        with torch.no_grad():
            logs = {
                "distill": loss_u.mean().item(),
                "v-branch": loss_v.mean().item(),
            }
        if self.dmd_weight == 0.0:
            # early exit
            return imf_loss.mean(), logs

        # DMD
        hat_x = z
        ts = torch.linspace(1, 0, self.unroll_step + 1).to(x)
        omega = torch.full_like(bt, self.precfg)
        t_min, t_max = self.cfgrange
        t_min = torch.full_like(bt, t_min)
        t_max = torch.full_like(bt, t_max)
        # sampling
        F_th = 0
        indices = torch.arange(bsize, device=x.device)[:, None, None, None]
        for i, (t_cur, t_next) in enumerate(zip(ts, ts[1:])):
            bt_cur, bt_next = t_cur.repeat(bsize), t_next.repeat(bsize)
            u, _ = model(hat_x, bt_cur, bt_cur - bt_next, omega, t_min, t_max, y=c)
            hat_x = hat_x + (t_next - t_cur) * u.detach()
            # drop gradient partially
            if not self.unroll_fullprop:
                u = torch.where(
                    torch.logical_and(
                        i * bsize // self.unroll_step <= indices,
                        indices < (i + 1) * bsize // self.unroll_step,
                    ),
                    u,
                    u.detach(),
                )
            F_th = F_th + (t_cur - t_next) * u

        # time-shifted t
        if self.gamma:
            t = torch.rand_like(t)
            t = self.gamma * t / ((self.gamma - 1) * t + 1)
            bt = t.flatten()

        # guiding network
        assert ema is not None
        guide = ema if teacher is None else teacher

        if self.dmd_indep_z:
            z = torch.randn_like(z)
        hat_x_t = (1 - t) * hat_x + t * z
        masked = omega.masked_fill(torch.logical_or(bt < t_min, bt > t_max), 1.0)
        with torch.no_grad():
            _, teacher = guide(hat_x_t, bt, zeros, masked, zeros, ones, c)
            fake, _ = ema(hat_x_t, bt, bt, omega, t_min, t_max, c)
            grad = fake - teacher
            if self.dmd_ada_weight:
                grad = grad / (teacher - F_th).abs().mean(dim=(1, 2, 3), keepdim=True)
                grad = torch.nan_to_num(grad)

            dmd_target = F_th - grad

        dmd_loss = (F_th - dmd_target).square().mean(dim=(1, 2, 3))
        if self.dmd_huber:
            dmd_loss = (dmd_loss + self.dmd_huber**2).sqrt() - self.dmd_huber

        w = self.dmd_weight
        if self.dmd_cos_weight:
            w = w * (bt * 1.57).cos()

        with torch.no_grad():
            logs["dmd"] = dmd_loss.mean().item()

        loss = imf_loss + dmd_loss * w
        return loss.mean(), logs

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        model: nn.Module,
        c: torch.Tensor,
        sampling_steps: int = 2,
        precfg: torch.Tensor | None = None,
        cfg_start: torch.Tensor | None = None,
        cfg_end: torch.Tensor | None = None,
    ):
        device = z.device
        bsize, *_ = z.shape
        # assign default values
        if precfg is None:
            precfg = torch.tensor([self.precfg] * bsize).to(z)

        if cfg_start is None:
            cfg_start, _ = self.cfgrange
            cfg_start = torch.tensor([cfg_start] * bsize).to(z)

        if cfg_end is None:
            _, cfg_end = self.cfgrange
            cfg_end = torch.tensor([cfg_end] * bsize).to(z)

        # uniform sampling
        t_steps = torch.linspace(1.0, 0.0, sampling_steps + 1, device=device)

        x_cur = z
        samples = [x_cur.clone()]
        # assert isinstance(model, FlowMap)
        for t_cur, t_next in zip(t_steps, t_steps[1:]):
            bt, bs = t_cur.repeat(bsize), t_next.repeat(bsize)
            u, _ = model(x_cur, bt, bt - bs, precfg, cfg_start, cfg_end, y=c)
            x_cur = x_cur + (t_next - t_cur) * u
            samples.append(x_cur)

        m = torch.tensor(self.mean)[:, None, None].to(z)
        s = torch.tensor(self.std)[:, None, None].to(z)
        return torch.stack(samples, dim=0) * s + m
