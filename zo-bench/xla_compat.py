"""
PyTorch/XLA (TPU) support for ZO-Bench. Imported first by run.py.
"""
import json
import time

import torch

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.runtime as xr

    HAS_XLA = True
except ImportError:
    HAS_XLA = False

if HAS_XLA:
    # transformers 4.28 still calls these; torch_xla 2.9 moved them to torch_xla.runtime
    if not hasattr(xm, "xrt_world_size"):
        xm.xrt_world_size = xr.world_size
    if not hasattr(xm, "get_ordinal"):
        xm.get_ordinal = xr.global_ordinal


def _patch_opt_attention_clamp():
    """
    transformers 4.28 OPTAttention clamps fp16 scores with
        torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
    The 0-dim constant is float32. Eager PyTorch keeps the result in fp16, but torch_xla's maximum lowering yields
    f32 while the tensor still reports fp16, so the later .to(torch.float16) is skipped and the f32 values reach the
    fp16 fc1 ("mixed precision is disallowed"). Give the constant the scores' dtype: same value, same eager semantics.
    """
    import inspect
    import textwrap

    from transformers.models.opt import modeling_opt

    old = "torch.tensor(torch.finfo(attn_weights.dtype).min)"
    src = textwrap.dedent(inspect.getsource(modeling_opt.OPTAttention.forward))
    if old not in src:
        return
    src = src.replace(old, "torch.tensor(torch.finfo(attn_weights.dtype).min, dtype=attn_weights.dtype)")
    namespace = {}
    exec(compile(src, modeling_opt.__file__, "exec"), modeling_opt.__dict__, namespace)
    modeling_opt.OPTAttention.forward = namespace["forward"]


if HAS_XLA:
    _patch_opt_attention_clamp()


def on_xla(tensor):
    return tensor.device.type == "xla"


def no_grad_context(device):
    """torch.inference_mode() breaks views on XLA ("Cannot set version_counter for inference tensor"); no_grad there."""
    return torch.no_grad() if device.type == "xla" else torch.inference_mode()


def stream_seed(seed, index, salt=0):
    """
    Seed for the index-th parameter of one random stream. On XLA, torch.manual_seed does not reseed the device RNG,
    so every regeneration of z (and of the sparse-gradient mask) reseeds the XLA RNG explicitly per parameter.
    """
    return (int(seed) * 1000003 + index * 7919 + salt * 104729) % (2 ** 31 - 1)


def log_xla_metrics(path, step):
    """Append compile/execute counts and all aten:: (CPU fallback) counters to a jsonl file."""
    if not HAS_XLA:
        return
    compile_data = met.metric_data("CompileTime")
    execute_data = met.metric_data("ExecuteTime")
    record = {
        "time": time.time(),
        "step": step,
        "compile_count": compile_data[0] if compile_data else 0,
        "compile_time_s": compile_data[1] / 1e9 if compile_data else 0.0,
        "execute_count": execute_data[0] if execute_data else 0,
        "aten_counters": {k: met.counter_value(k) for k in met.counter_names() if k.startswith("aten::")},
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def write_xla_metrics_report(path):
    if HAS_XLA:
        with open(path, "w") as f:
            f.write(met.metrics_report())
