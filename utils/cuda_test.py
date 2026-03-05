import torch
import torchvision
print(f"Torch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device Count: {torch.cuda.device_count()}")
print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
print(f"CUDA Device Capability: {torch.cuda.get_device_capability(0)}")
print(f"CUDA Device Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")