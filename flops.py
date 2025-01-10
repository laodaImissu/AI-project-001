import torch
import torchvision
from thop import profile

#模型,优化器
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from Vgg16Net import Vgg16_net
model=Vgg16_net().to(device)

#给定一个input
input = torch.randn(1, 256, 8, 8)

flops, params = profile(model.layer3, (input,))

print(f"{model.layer3} FLOPs: {(flops / 1e6 ):.2f} M")
