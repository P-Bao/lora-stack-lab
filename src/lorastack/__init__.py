"""lora-stack-lab: composability study of LoRA improvements (GraLoRA + GoRA + AltLoRA + BA-LoRA)."""
import warnings

# HF gradient checkpointing warns on every no-grad forward (eval, base/W0 reference
# forward). Harmless and expected there; our own _ckpt guard falls back instead of
# warning, so this message is pure noise in the training logs.
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")

__version__ = "0.1.0"
