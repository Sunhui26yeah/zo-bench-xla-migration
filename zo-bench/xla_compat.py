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


def on_xla(tensor):
    return tensor.device.type == "xla"


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
