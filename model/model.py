import torch
import torchvision.models as models

# Загружаем предобученную модель
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, 10)  # CIFAR-10 имеет 10 классов
torch.save(model.state_dict(), "cifar10_model.pth")