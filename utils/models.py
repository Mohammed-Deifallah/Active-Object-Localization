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

class DQN(nn.Module):
    def __init__(self, f_extr_name='vgg16'):
        super(DQN, self).__init__()
        f_extr_dim = 25088 if f_extr_name=='vgg16' else 512*4 #resnet50
        self.classifier = nn.Sequential(
            nn.Linear( in_features= 81 + f_extr_dim, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=9)
        )
    def forward(self, x):
        return self.classifier(x)