import os
import torchvision.transforms as transforms
import torchvision


def get_transform(train):
    """
        Permettant la préparation d'une fonction normalisation + le redimensionnement des images du jeu de données.
    """
    base_size = 520
    crop_size = 480

    min_size = int((0.5 if train else 1.0) * base_size)
    max_size = int((2.0 if train else 1.0) * base_size)
    transf = []
    transf.append( transforms.ToTensor())
    transf.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transf)
    
def make_image_transform(image_transform_params: dict,
                         transform: object):
    resize_image = image_transform_params['image_mode']
    if resize_image == 'none':
        preprocess_image = None
    elif resize_image == 'shrink':
        preprocess_image = transforms.Resize((image_transform_params['output_image_size']['width'],
                                              image_transform_params['output_image_size']['height']))
    elif resize_image == 'crop':
        preprocess_image = transforms.CenterCrop((image_transform_params['output_image_size']['width'],
                                                  image_transform_params['output_image_size']['height']))

    if preprocess_image is not None:
        if transform is not None:
            image_transform = transforms.Compose([preprocess_image, transform])
        else:
            image_transform = preprocess_image
    else:
        image_transform = transform
    return image_transform


def read_voc_dataset(path, year):
    T = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            ])
    voc_data =  torchvision.datasets.VOCDetection(path, year=year, image_set='train', transform=T)
    voc_val =  torchvision.datasets.VOCDetection(path, year=year, image_set='val', transform=T)

    return voc_data, voc_val

def get_images_labels(dataloader):
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    return images, labels