"""Step 1 — Verify GPU / CUDA availability."""
import subprocess, sys


def run() -> bool:
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"PyTorch {torch.__version__}  |  CUDA: {cuda_ok}")
        if cuda_ok:
            print(f"GPU : {torch.cuda.get_device_name(0)}")
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"VRAM: {total:.1f} GB")
        else:
            print("WARNING: CUDA not available — models will run on CPU (slow).")
        return cuda_ok
    except ImportError:
        print("ERROR: PyTorch not found. Run: pip install torch")
        return False
