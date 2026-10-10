import torch.nn as nn
from torchvision import models

def get_mobilenet_v2(num_classes=10):
    # 1. Load the pre-trained model and its ImageNet weights
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # 2. Freeze the base model (Feature Extractor)
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. Swap the classification head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model


def get_squeezenet(num_classes=10):
    # 1. Load the pre-trained SqueezeNet model
    model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT)
    
    # 2. Freeze the base model
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. Swap the classification head
    # SqueezeNet uses a 1x1 Convolution instead of a Linear layer for its output
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model.num_classes = num_classes
    
    return model