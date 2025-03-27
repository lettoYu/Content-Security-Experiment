import torch
print(torch.__version__)            # 应该输出 2.1.0+cu118
print(torch.cuda.is_available())    # 应该输出 True
print(torch.cuda.get_device_name(0))  # 应输出你的 RTX 3060 名称
