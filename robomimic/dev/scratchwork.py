import os
import torch

CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")

avail = torch.cuda.is_available()
counts = torch.cuda.device_count()
current = torch.cuda.current_device()
name = torch.cuda.get_device_name(0)

print()