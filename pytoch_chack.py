import torch

if torch.cuda.is_available():
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"CUDA version used by PyTorch: {torch.version.cuda}")
    print(f"cuDNN version used by PyTorch: {torch.backends.cudnn.version()}")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. PyTorch will run on CPU.")