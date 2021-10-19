use_cuda = True
import torch
import torch.nn as nn
from collections import namedtuple
import torchvision.transforms as transforms
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor
SAVE_MODEL_PATH = './models/q_network'
if use_cuda:
    criterion = nn.MSELoss().cuda()   
else:
    criterion = nn.MSELoss()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
])