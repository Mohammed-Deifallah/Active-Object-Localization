# Active Object Localization

This repository represents the final project of [Reinforcement Learning Course](http://files.skoltech.ru/data/edu/syllabuses/2021/MA060422.pdf?v=xj03xr) from [Skoltech University](https://www.skoltech.ru/en/). It tackles a non-RL problem using Deep Reinforcement Learning. This project is mainly based on [Active Object Localization with Deep Reinforcement Learning](https://arxiv.org/abs/1511.06015)

## Dataset

We used [PASCAL VOC 2007 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/), a well-known dataset for object recognition. This dataset contains various images for 20 different classes, spanning from human beings and living creatures to vehicles and indoor objects.
For the sake of our academic project, we trained the training set on a less number of classes, and used the validation set for testing.

## Metrics

Referring to the below-mentioned original paper, we used **AP** (Average Precision) as our accuracy metric, side by side to **Recall**.

## Acknowledgements

Frankly speaking, we would like to thank [Rayan Samy](https://github.com/rayansamy/) for being our consultant as this project is inspired by [his repository](https://github.com/rayansamy/Active-Object-Localization-Deep-Reinforcement-Learning).
