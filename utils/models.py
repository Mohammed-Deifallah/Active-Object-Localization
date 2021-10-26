import torch.nn as nn
import torchvision

class FeatureExtractor(nn.Module):
    def __init__(self, network='vgg16'):
        super(FeatureExtractor, self).__init__()
        if network == 'vgg16':
            model = torchvision.models.vgg16(pretrained=True)
            self.features = model.features
        elif network == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True)
            model.fc = nn.Identity()
            self.features = model
        else:
            model = torchvision.models.alexnet(pretrained=True)
            self.features = list(model.children())[0]

        model.eval() # to not do dropout
    def forward(self, x):
        x = self.features(x)
        return x