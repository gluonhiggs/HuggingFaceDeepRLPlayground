import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.current_device())  # Should print 0 (or your GPU index)
print(torch.cuda.get_device_name(0))  # Should print your GPU name, e.g., "NVIDIA GeForce RTX 3080"