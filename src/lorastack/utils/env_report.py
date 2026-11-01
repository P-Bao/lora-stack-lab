"""Environment report: versions + GPU, appended to every result JSON."""
from __future__ import annotations

import platform


def env_report() -> dict:
    import torch
    import transformers
    import peft
    import datasets

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "datasets": datasets.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        info["gpu_vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    return info
