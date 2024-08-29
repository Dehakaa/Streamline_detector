# Extracting Quantitative Streamline Information from Surface Flow Visualization Images in a Linear Cascade using Convolutional Neural Networks.

These files contain the code for Paper Extracting Quantitative Streamline Information from Surface Flow Visualization Images in a Linear Cascade using Convolutional Neural Networks. 

I developed it on Linux, so it might be a bit problematic on Windows. Don't try to train without a GPU

The files are provided in the hope that they will be useful to other researchers but we provide the software ``AS IS'' WITHOUT WARRANTY OF ANY KIND.  The code is meant to be a demonstration of the techniques used in the paper. I think my idea might be good but my coding skill is really terrible. Thus, do some modification according you task, and hopefully it might work.

## Requirements

* [Python 3.7](https://www.python.org/downloads/release/python-370/g)
* [Pytorch >=1.4](https://pytorch.org/) (Last test 1.9)
* [OpenCV](https://pypi.org/project/opencv-python/)
* [Matplotlib](https://matplotlib.org/3.1.1/users/installing.html)
* [Kornia](https://kornia.github.io/)
* Other package like Numpy, h5py, PIL, json, skimage. 


## Project Architecture

```
Camera Lens Calibration -> contains the calibration images and the python code for the lens correction and coordinate detection.
train.py -> train the egde detection neural networks
test.py -> test the egde detection neural networks
preprocessing.py -> including calibrated visualization image, division into cells, grayscale conversion, image smoothing, and exposing intensity gradients. 
postprocessing.py -> get your flow direction result
```

Before to start please check dataset.py, from the first line of code you can see the datasets used for training/testing. Before you generate your own result, please check again.

## Datasets used for Training

Edge detection datasets
* [BIPED](https://xavysp.github.io/MBIPED/)

## How to use

1. After completing the setup, use `preprocessing.py` to preprocess the images to be processed.
2. Download the dataset and check the Parser settings, then run `train.py`.
3. Choose the appropriate stride and neighborhood size (they need to match the size of the images), then run `postprocessing.py`.

Tips: If you do not want to train a neural network from scratch, or if your device does not have sufficient performance, you can also use the results of Canny edge detection to run `postprocessing.py`.

## Acknowledgement

* We like to thanks to the previous repo: [LDC](https://GitHub.com/xavysp/LDC)

