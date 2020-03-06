## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, we will use deep neural networks and convolutional neural networks to classify traffic signs.
The model will be trained and validated on traffic signs images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained , we will test the model on images of German traffic signs that we fetch online ( not from the dataset).

You can view the Traffic sign classifier project [here](https://github.com/asagar60/Udacity-SDCND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier-v2.0.ipynb)

Detailed writeup of the project explaining the thought process involving in how I trained the model can be viewed [here](https://github.com/asagar60/Udacity-SDCND-Traffic-Sign-Classifier-Project/blob/master/writeup%20-v2.0.md)

**Installing Dependencies**
---

- opencv           -  `pip install opencv-python`
- pandas           - `pip install pandas`
- Tensorflow - GPU - `conda install tensorflow-gpu`
- matplotlib       - `pip install matplotlib`


[//]: # (Image References)

[image1]: ./writeup_images-v2.0/bar_chart_training_data.JPG "Visualization of Training Set"
[image15]: ./writeup_images-v2.0/lenet-5.JPG "lenet-5"




The Project
---
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


### Load the data set

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

### Explore, summarize and visualize the data set

![Distribution of the dataset][image1]


### Design, train and test a model architecture

![lenet-5][image15]

LeNet-5 is a very simple network. It only has 7 layers, among which there are 3 convolutional layers (C1, C3 and C5), 2 sub-sampling (pooling) layers (S2 and S4), and 2 fully connected layer (F6), that are followed by the output layer. Convolutional layers use 5 by 5 convolutions with stride 1.

* training set accuracy of 97.6%
* validation set accuracy of 97.7%
* test set accuracy of 94.8%

### Use the model to make predictions on new images

The model was able to correctly guess 7 of the 10 traffic signs
