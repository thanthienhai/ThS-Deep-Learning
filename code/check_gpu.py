import torch
import sys

with open('gpu_status.txt', 'w') as f:
    f.write(f"Python version: {sys.version}\n")
    f.write(f"PyTorch version: {torch.__version__}\n")
    f.write(f"CUDA available: {torch.cuda.is_available()}\n")
    if torch.cuda.is_available():
        f.write(f"CUDA device count: {torch.cuda.device_count()}\n")
        f.write(f"Current device: {torch.cuda.current_device()}\n")
        f.write(f"Device name: {torch.cuda.get_device_name(0)}\n")
    else:
        f.write("CUDA is NOT available. Training will run on CPU.\n")
