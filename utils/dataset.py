import os
import torchvision.transforms as transforms
import torchvision


def read_voc_dataset(path, year):
    T = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            ])
    voc_data =  torchvision.datasets.VOCDetection(path, year=year, image_set='train', transform=T)
    voc_val =  torchvision.datasets.VOCDetection(path, year=year, image_set='val', transform=T)

    return voc_data, voc_val