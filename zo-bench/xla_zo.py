"""
Flat-buffer ZO-SGD parameter sweeps for PyTorch/XLA.

In lazy mode every ZO step re-traces four sweeps over all parameter tensors (perturb +eps, perturb -2eps, restore,
update), about 4,300 small ops per sweep, and the step is host-bound. Here all trainable parameters become views of
one flat buffer, so each sweep is a handful of ops on that buffer:
    z     = N(0, 1) of length numel, in the parameter dtype      (XLA RNG seeded once per sweep with zo_seed)
    mask  = N(0, 1) sample (parameter dtype) < per-element ratio  (seeded with sparse_seed), z[mask] = 0
which is the distribution of the original per-tensor torch.normal + fast_random_mask_like. The same seeds reproduce
the same z and mask in all four sweeps.

The fp16 arithmetic of the original eager code is reproduced op by op (compute in fp32, round to the dtype):
    perturb: param + scaling_factor * z * zo_eps            (three rounded ops)
    update:  grad = projected_grad * z;  SGD(momentum=0, weight_decay=0): param.add_(grad, alpha=-lr)
"""
import torch
import torch_xla.core.xla_model as xm

from xla_compat import stream_seed


class XlaZoSweeps:

    def __init__(self, named_params, grad_sparsity_fn, zo_eps):
        self.params = [p for _, p in named_params]
        dtype, device = self.params[0].dtype, self.params[0].device
        assert all(p.dtype == dtype for p in self.params)
        self.numel = sum(p.numel() for p in self.params)
        self.zo_eps = zo_eps
        with torch.no_grad():
            self.flat = torch.cat([p.data.reshape(-1) for p in self.params])
            offset = 0
            for p in self.params:
                n = p.numel()
                p.data = self.flat[offset:offset + n].view(p.shape)
                offset += n
            ratios = [grad_sparsity_fn(n) for n, _ in named_params]
            if all(r is None for r in ratios):
                self.ratio = None
            else:
                # the original compares the dtype mask sample with the Python ratio in that dtype
                self.ratio = torch.cat([torch.full((p.numel(),), float("-inf") if r is None else r, dtype=dtype,
                                                   device=device) for r, p in zip(ratios, self.params)])
        xm.mark_step()

    def _z(self, zo_seed, sparse_seed):
        xm.set_rng_state(stream_seed(zo_seed, 0))
        z = torch.normal(mean=0, std=1, size=(self.numel,), device=self.flat.device, dtype=self.flat.dtype)
        if self.ratio is not None:
            xm.set_rng_state(stream_seed(sparse_seed, 0, salt=1))
            mask_sample = torch.empty_like(z).normal_()
            z = z.masked_fill(mask_sample < self.ratio, 0)
        return z

    @torch.no_grad()
    def perturb(self, scaling_factor, zo_seed, sparse_seed):
        z = self._z(zo_seed, sparse_seed)
        t = (z.float() * scaling_factor).to(z.dtype)
        t = (t.float() * self.zo_eps).to(z.dtype)
        self.flat.copy_((self.flat.float() + t.float()).to(z.dtype))
        xm.mark_step()

    @torch.no_grad()
    def update(self, projected_grad, lr, zo_seed, sparse_seed):
        z = self._z(zo_seed, sparse_seed)
        grad = (z.float() * projected_grad).to(z.dtype)
        self.flat.copy_((self.flat.float() - lr * grad.float()).to(z.dtype))
        xm.mark_step()
