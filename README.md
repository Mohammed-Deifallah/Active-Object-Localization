# Active Object Localization

This repository represents the final project of [Reinforcement Learning Course](http://files.skoltech.ru/data/edu/syllabuses/2021/MA060422.pdf?v=xj03xr) from [Skoltech University](https://www.skoltech.ru/en/). It tackles a non-RL problem using Deep Reinforcement Learning. This project is mainly based on [Active Object Localization with Deep Reinforcement Learning](https://arxiv.org/abs/1511.06015)

## Git branches
- `main` - implementation according to original article (Pytorch+DQN) - based on Rayan Samy's code referenced below
- `resnet50` - same as main but feature extractor is pre-trained resnet50 (instead of vgg16 used in main)
- `sb3_gym` - experimental impementation based on PPO algorithm from 'Stable Baselines 3' library (feature extractor is still pre-trained CNN)

## Code Overview (main branch)
- `Training.ipynb` is used to reproduce the training process of the model.
- `Testing.ipynb` is used to reproduce the testing process of the model and visualize some examples of localization.
- `Plotting.ipynb` is used to plot all graphs and charts shown above using `media` folder.
- `media` is a folder to save all examples of localization and graphs.
- `models` is a directory which is needed to keep the saved models weights after training. This is an [example](https://drive.google.com/drive/folders/1JvwQquAVerxj7DD-qtKJGvmc_VUNBpr8?usp=sharing) of the so-called folder.
- `utils` is a directory contaning the following files:
  - `agent.py`: a wrapper for the per-class agent that contains the whole components of RL (ϵ-greedy policy, reward, ... etc).
  - `models.py`: a wrapper for the two main modules of the network: _Feature Extractor_ and _DQN_.
  - `dataset.py`: a separate file for reading the dataset (train and val).
  - `tools.py`: a collection of useful functions, such as computing the metrics or extracting a specified class from the dataset.

## Model

The following figure shows the high-level diagram of the used DQN model architecture from the authors of the original work:

<p align='center'><img src='media/model.JPG' style="width: 650px; max-width: 100%; height: auto" title="Click to enlarge picture"/></p>

According to the paper, we used [VGG-16](https://pytorch.org/vision/stable/models.html#torchvision.models.vgg16) model as our pre-trained CNN on [ImageNet](https://www.image-net.org/) dataset.

## Dataset

We used [PASCAL VOC 2007 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/), a well-known dataset for object recognition. This dataset contains various images for 20 different classes, spanning from human beings and living creatures to vehicles and indoor objects.
For the sake of our academic project, we trained the training set on a less number of classes, and used the validation set for testing.

P.S. The current version of the code only supports the offline dataset because the (mentioned) official website of the dataset was down as described at the start of `training.ipynb`.

## Metrics

Referring to the above-mentioned original paper, we used **AP** (Average Precision) as our accuracy metric, side by side to **Recall**.

## Results

Limited by computational resources, the model shown above is trained on only 5 classes. The following two charts show the performance of the model according to different values of the threshold of IoU:

Average Precision             |  Recall
:-------------------------:|:-------------------------:
<img src='media/VGG16 Model - AP.png' alt='AP' title="Click to enlarge picture"/>  |  <img src='media/VGG16 Model - Recall.png' alt='Recall' title="Click to enlarge picture"/>

In addition to those charts, there's another humble chart to demonstrate the comparison to the original results as follows:

<p align='center'><img src="media/comparison.png" style="width: 650px; max-width: 100%; height: auto" title="Click to enlarge picture"/></p>

Kindly note that the model under study is trained using less data, and is also tested against a different dataset.

## Acknowledgements

Frankly speaking, we would like to thank [Rayan Samy](https://github.com/rayansamy/) for being our consultant as this project is inspired by [his repository](https://github.com/rayansamy/Active-Object-Localization-Deep-Reinforcement-Learning).
