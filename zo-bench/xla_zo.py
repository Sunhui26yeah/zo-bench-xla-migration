"""
Compiled ZO-SGD parameter sweeps for PyTorch/XLA.

In lazy mode every ZO step re-traces four sweeps over all parameter tensors (perturb +eps, perturb -2eps, restore,
update), about 3,100 small ops, and the step is host-bound. Here each sweep is one function compiled once with
torch.compile(backend="openxla") and replayed every step.

z and the sparse-gradient mask are generated statelessly from (seed, global element index) with a 32-bit integer hash
and Box-Muller, so the same seed reproduces the same z and mask in all four sweeps without RNG state. The
distributions match the original: z ~ N(0, 1) rounded to the parameter dtype, and an element is masked when a second,
independent N(0, 1) sample (in the parameter dtype) is below the tensor's sparsity value, as in fast_random_mask_like.

The fp16 arithmetic of the original eager code is reproduced op by op (compute in fp32, round to the parameter
dtype):
    perturb: param + scaling_factor * z * zo_eps    (three rounded ops)
    update:  grad = projected_grad * z;  SGD(momentum=0, weight_decay=0): param.add_(grad, alpha=-lr)
"""
import math

import torch

_MUL_A = 0x7FEB352D
_MUL_B = 0x846CA68B - (1 << 32)  # as signed int32
_GOLDEN = 0x9E3779B1 - (1 << 32)


def _hash32(x):
    """lowbias32 integer hash on int32 tensors (wrapping multiply, logical shifts)."""
    x = x ^ ((x >> 16) & 0xFFFF)
    x = x * _MUL_A
    x = x ^ ((x >> 15) & 0x1FFFF)
    x = x * _MUL_B
    x = x ^ ((x >> 16) & 0xFFFF)
    return x


def _normal(index, seed_key):
    """Standard normal (fp32) for each int32 global element index under an int32 seed key (Box-Muller)."""
    h = _hash32(index * _GOLDEN + seed_key)
    h1 = _hash32(h ^ 0x68E31DA4)
    h2 = _hash32(h ^ 0x1B56C4E9)
    u1 = (((h1 >> 8) & 0xFFFFFF).float() + 0.5) * (1.0 / (1 << 24))
    u2 = ((h2 >> 8) & 0xFFFFFF).float() * (1.0 / (1 << 24))
    return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos((2.0 * math.pi) * u2)


class XlaZoSweeps:
    """Compiled perturbation and SGD-update sweeps over a fixed list of parameters."""

    def __init__(self, named_params, grad_sparsity_fn, zo_eps):
        self.params = [p for _, p in named_params]
        # the original compares the fp16 mask sample with the Python ratio in fp16, i.e. with the ratio rounded to fp16
        self.sparsity = [None if grad_sparsity_fn(n) is None else float(torch.tensor(grad_sparsity_fn(n), dtype=p.dtype))
                         for n, p in named_params]
        self.offsets, total = [], 0
        for p in self.params:
            self.offsets.append(total)
            total += p.numel()
        assert total < 2 ** 31, "global element index must fit in int32"
        self.zo_eps = zo_eps
        self._perturb = {}
        self._update = torch.compile(self._update_fn, backend="openxla", fullgraph=True, dynamic=False)

    def _seed_keys(self, zo_seed, sparse_seed):
        device = self.params[0].device
        keys = torch.tensor([_hash32_int(zo_seed), _hash32_int(sparse_seed ^ 0x5BD1E995)], dtype=torch.int32)
        return keys.to(device)

    def _z(self, i, p, keys):
        index = torch.arange(p.numel(), dtype=torch.int32, device=p.device) + self.offsets[i]
        z = _normal(index, keys[0]).to(p.dtype)
        if self.sparsity[i] is not None:
            mask_normal = _normal(index, keys[1]).to(p.dtype)
            z = z.masked_fill(mask_normal < self.sparsity[i], 0)
        return z.view(p.shape)

    def _perturb_fn(self, params, keys, scaling_factor):
        out = []
        for i, p in enumerate(params):
            z = self._z(i, p, keys)
            t = (z.float() * scaling_factor).to(p.dtype)
            t = (t.float() * self.zo_eps).to(p.dtype)
            out.append((p.float() + t.float()).to(p.dtype))
        return out

    def _update_fn(self, params, keys, projected_grad, lr):
        out = []
        for i, p in enumerate(params):
            z = self._z(i, p, keys)
            grad = (projected_grad * z.float()).to(p.dtype)
            out.append((p.float() - lr * grad.float()).to(p.dtype))
        return out

    def perturb(self, scaling_factor, zo_seed, sparse_seed):
        if scaling_factor not in self._perturb:
            fn = lambda params, keys, s=float(scaling_factor): self._perturb_fn(params, keys, s)
            self._perturb[scaling_factor] = torch.compile(fn, backend="openxla", fullgraph=True, dynamic=False)
        new = self._perturb[scaling_factor]([p.data for p in self.params], self._seed_keys(zo_seed, sparse_seed))
        for p, n in zip(self.params, new):
            p.data = n

    def update(self, projected_grad, lr, zo_seed, sparse_seed):
        device = self.params[0].device
        g = torch.tensor(projected_grad, dtype=torch.float32).to(device)
        lr_t = torch.tensor(lr, dtype=torch.float32).to(device)
        new = self._update([p.data for p in self.params], self._seed_keys(zo_seed, sparse_seed), g, lr_t)
        for p, n in zip(self.params, new):
            p.data = n


def _hash32_int(x):
    """The same lowbias32 hash on a Python int (host side, for seed keys), returned as a signed int32 value."""
    m = 0xFFFFFFFF
    x &= m
    x ^= x >> 16
    x = (x * 0x7FEB352D) & m
    x ^= x >> 15
    x = (x * 0x846CA68B) & m
    x ^= x >> 16
    return x - (1 << 32) if x >= (1 << 31) else x
