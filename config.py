cloud = 'kaggle'
use_cuda = True

import os
import torch
import torch.nn as nn
from collections import namedtuple
import torchvision.transforms as transforms
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

# CLASSES = ['cat', 'cow', 'dog', 'bird', 'car']
CLASSES = ['person']
BEST_MODELS = ['models/Oct26__22_26/person/best_model.zip']
STATS_PATHS = ['models/Oct26__22_26/person/Oct27__05_07.pkl']

# CLASSES = ['cat']
# BEST_MODELS = ['models/BEST/CAT/best_model.zip']
# STATS_PATHS = ['models/BEST/CAT/cat_2021Oct25_06_08.pkl']

SAVE_MODEL_PATH = './models/q_network'
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

MEDIA_ROOT = './media'
os.makedirs(MEDIA_ROOT, exist_ok=True)

VOC2007_ROOT = "../data"
if cloud == 'kaggle':
    VOC2007_ROOT = "/kaggle/input/pascal-voc-2007"
if cloud == 'colab':
    VOC2007_ROOT = "/contents/data/pascal-voc-2007"

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