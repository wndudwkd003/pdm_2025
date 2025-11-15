import torch

print("PyTorch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("cuDNN available:", torch.backends.cudnn.is_available())
print("cuDNN enabled:", torch.backends.cudnn.enabled)
print("cuDNN version (int):", torch.backends.cudnn.version())  # 예: 91200 -> 9.12.0
print("GPU:", torch.cuda.get_device_name(0))
