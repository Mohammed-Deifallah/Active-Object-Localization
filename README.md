# Active Object Localization

This repository represents the final project of [Reinforcement Learning Course](http://files.skoltech.ru/data/edu/syllabuses/2021/MA060422.pdf?v=xj03xr) from [Skoltech University](https://www.skoltech.ru/en/). It tackles a non-RL problem using Deep Reinforcement Learning. This project is mainly based on [Active Object Localization with Deep Reinforcement Learning](https://arxiv.org/abs/1511.06015)

## Model

The following figure shows the high-level diagram of the used DQN model architecture from the authors of the original work:

<a href="https://drive.google.com/uc?export=view&id=1MTDMU6gMGQ_g_HPBBjp8_1AeTV14RV5F"><p align="center"><img src="https://drive.google.com/uc?export=view&id=1MTDMU6gMGQ_g_HPBBjp8_1AeTV14RV5F" style="width: 650px; max-width: 100%; height: auto" title="Click to enlarge picture"/></p></a>

According to the paper, we used [VGG-16](https://pytorch.org/vision/stable/models.html#torchvision.models.vgg16) model as our pre-trained CNN on [ImageNet](https://www.image-net.org/) dataset.

## Dataset

We used [PASCAL VOC 2007 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/), a well-known dataset for object recognition. This dataset contains various images for 20 different classes, spanning from human beings and living creatures to vehicles and indoor objects.
For the sake of our academic project, we trained the training set on a less number of classes, and used the validation set for testing.

## Metrics

Referring to the below-mentioned original paper, we used **AP** (Average Precision) as our accuracy metric, side by side to **Recall**.

## Results

Limited by computational resources, the model shown above is trained on only 5 classes. The following two charts show the performance of the model according to different values of the threshold of IoU:

Average Precision             |  Recall
:-------------------------:|:-------------------------:
![AP](https://drive.google.com/uc?export=view&id=1-NL9fCFm36Jvpl-tlm36uKCqLfbP3rNF)  |  ![Recall](https://drive.google.com/uc?export=view&id=1-NBD9kaQ4LhvxC3QXEfUpvJIVM2diZn-)

In addition to those charts, there's another humble chart to demonstrate the comparison to the original results as follows:

<a href="https://drive.google.com/uc?export=view&id=1-No1TMxnOrUzDFj24vpMsLx2-hMPDcAV"><img src="https://drive.google.com/uc?export=view&id=1-No1TMxnOrUzDFj24vpMsLx2-hMPDcAV" style="width: 650px; max-width: 100%; height: auto" title="Click to enlarge picture"/></a>

Kindly note that the model under study is trained using less data, and is also tested against a different dataset.

## Acknowledgements

Frankly speaking, we would like to thank [Rayan Samy](https://github.com/rayansamy/) for being our consultant as this project is inspired by [his repository](https://github.com/rayansamy/Active-Object-Localization-Deep-Reinforcement-Learning).
