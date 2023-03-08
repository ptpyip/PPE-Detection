# Real-time PPE Detection with Deep Learning

This repo contains some code we used for the final project of COMP4471 Deep Learning with Computer Vision. We also include some notebook we used to train our model on Google Colab.

## Abstract 
In this project, we investigated the methods for real-time PPE detection.
We fine-tuned and compared the performance of 3 object detection model: 
Faster-R-CNN, SSD, and YOLO.

## Paper
[Real-time PPE Detection with Deep Learning](COMP4471_Project.pdf)

## Demo
We use our fine-tuned YOLOv5 model to detect whether workers are wearing helmet or not, using a clip captured from YouTube.
![YOLO_detect ](https://user-images.githubusercontent.com/18398848/223604593-7616b1b8-c64d-4d59-ae1b-baee448f1381.gif)


## Acknowledgements
Here are some repo inspired our work:
- [Nested PPE detection FasterRCNN](https://github.com/mohammadakz/Nested_PPE_detection_FasterRCNN)

- [Real-time PPE detection](https://github.com/ZijianWang1995/PPE_detection)

- [reflective-clothes-detect-yolov5](https://github.com/gengyanlei/reflective-clothes-detect-yolov5)
    (Our training pipe line are modified with its `train.py`)

We also greatly rely on [GluonCV Tutorial](https://cv.gluon.ai/tutorials/index.html), [Pytorch Tutorials](https://pytorch.org/tutorials/), and [torchvision Doc](https://pytorch.org/vision/stable/index.html) to complete our training.

## Dataset
- [Safety-Helmet-Wearing-Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)

## What I Learn
