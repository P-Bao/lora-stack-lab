# Environment gate: run first in every experiment (python tests/test_env.py).
# exit(1) on any mismatch so a run never starts on a drifted env.
import sys

import peft
import transformers

PEFT_PIN = "0.21.2"
TRANSFORMERS_PIN = "4.57.6"


def main() -> int:
    ok = True
    if peft.__version__ != PEFT_PIN:
        print(f"FAIL: peft {peft.__version__} != pinned {PEFT_PIN}")
        ok = False
    if transformers.__version__ != TRANSFORMERS_PIN:
        print(f"FAIL: transformers {transformers.__version__} != pinned {TRANSFORMERS_PIN}")
        ok = False

    import torch

    print(f"python      {sys.version.split()[0]}")
    print(f"torch       {torch.__version__}")
    print(f"cuda        {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        total_vram = 0.0
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram = props.total_memory / 1024**3
            total_vram += vram
            print(f"gpu{i}       {props.name} {vram:.1f} GiB")
        print(f"vram_total  {total_vram:.1f} GiB")
    else:
        # GPU run: no CUDA is a broken env for this lab.
        print("FAIL: CUDA not available (runs require T4 x2)")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
