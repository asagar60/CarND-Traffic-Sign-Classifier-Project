# **Traffic Sign Recognition**

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images-v2.0/bar_chart_training_data.JPG "Visualization of Training Set"
[image2]: ./writeup_images-v2.0/bar_chart_augmented_training_data.JPG "Visualization of Augmented Training Set"
[image3]: ./writeup_images-v2.0/Blurred_image.JPG "Image Blur"
[image4]: ./writeup_images-v2.0/Alter_brightness.JPG "Altered brightness"
[image5]: ./writeup_images-v2.0/Rotated_Image.JPG "Rotated Image"
[image6]: ./writeup_images-v2.0/preprocess_image.JPG "Preprocessed Image"
[image7]: ./writeup_images-v2.0/sample_images_training_set_cropped.jpg "training set"
[image8]: ./writeup_images-v2.0/Augmented_training_set_cropped.jpg "Augmented training set"
[image9]: ./writeup_images-v2.0/Accuracy_graph.JPG "Accuracy Graph"
[image10]: ./writeup_images-v2.0/loss_graph.JPG "Loss Graph"
[image11]: ./writeup_images-v2.0/preprocess_example_data.JPG "preprocess example data"
[image12]: ./writeup_images-v2.0/conv_layer_1.JPG "conv layer 1"
[image13]: ./writeup_images-v2.0/conv_layer_2.JPG "conv layer 2"
[image14]: ./writeup_images-v2.0/softmax_prob.JPG "softmax prob"
[image15]: ./writeup_images-v2.0/lenet-5.JPG "lenet-5"

[image16]: ./writeup_images-v2.0/traffic_sign/1.JPG "Slippery road"
[image17]: ./writeup_images-v2.0/traffic_sign/2.JPG "Roundabout mandatory"
[image18]: ./writeup_images-v2.0/traffic_sign/3.JPG "Bumpy road"
[image19]: ./writeup_images-v2.0/traffic_sign/4.JPG "Road work"
[image20]: ./writeup_images-v2.0/traffic_sign/5.JPG "Children crossing"
[image21]: ./writeup_images-v2.0/traffic_sign/6.JPG "Speed limit (50km/h)"
[image22]: ./writeup_images-v2.0/traffic_sign/7.JPG "Speed limit (30km/h)"
[image23]: ./writeup_images-v2.0/traffic_sign/8.JPG "Yield"
[image24]: ./writeup_images-v2.0/traffic_sign/9.JPG "Go straight or right"
[image25]: ./writeup_images-v2.0/traffic_sign/10.JPG "Turn right ahead"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed in the Training Set

![Distribution of the dataset][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

## Sample Set before Augmentation

![image7]


I decided to generate additional data because of the following reason(s):
- As shown in the Distribution of the data in the dataset in previous image, classes such as [0,19,37,41,42] have very few data. This might cause the model to be biased around other data .
Unbalanced dataset results in biased prediction.

- low variation in the image set .
i.e. for the model to perform optimally on any image , it should be trained on augmented dataset which will have rotated image, low brightness image.

other examples : Translated Image (partially visible Image), Zoomed in Image, cropped Image, high contrast Image etc.

Firstly , I generated all the augmented Image , such that all classes have almost same number of data points.

## Augmentation Steps

* Blurring the Image.

    ![Image Blur][image3]

* Change the brightness of the Image.

    ![Altered Brightness][image4]

* Rotate the Image.

    ![Rotated Image][image5]

## Sample Set after Augmentation


![Augmented_training_set_cropped][image8]


## Preprocessing all Images in the set

* Convert the Image to grayscale.
* Contrast-limited Adaptive Histogram Equalization.
* Normalize the pixel values [0. - 1.]

![Preprocessed Image][image6]


The difference between the original data set and the augmented data set is the following ,

* Augmented data set contains blurred Image , Altered Brightness Image , Rotated Image, and Combinations of them
* Augmented data set contains 46888 more images than the original Training Set

I decided to augment the dataset in such a way that all classes have almost same number of data points

Distribution of Images in Training Dataset after Image Augmentation

![Visualization of Augmented Training Set][image2]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My Lenet model consisted of the following layers:

|          Layer        |              Description	        		    |
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   					|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten               | outputs 400                                   |
| Fully connected       | outputs 120                                   |
| RELU					|												|
| Dropout          	    | rate = 0.5      								|
| Fully connected       | outputs 84                                    |
| RELU					|												|
| Dropout          	    | rate = 0.5      								|
| Fully connected		| outputs 43        							|
| Softmax				|            									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used

| Parameters      | Values         |
|:----------------|:--------------:|
| optimizer       | AdamOptimizer  |
| learning_rate   | 0.0005         |
| EPOCHS          | 20             |
| BATCH_SIZE      | 128            |
| beta            | 0.001          |

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 97.6%
* validation set accuracy of 97.7%
* test set accuracy of 94.8%

![model Accuracy][image9]

![loss accuracy][image10]

If an iterative approach was chosen:

**What was the first architecture that was tried and why was it chosen?**

![lenet-5][image15]

LeNet-5 is a very simple network. It only has 7 layers, among which there are 3 convolutional layers (C1, C3 and C5), 2 sub-sampling (pooling) layers (S2 and S4), and 2 fully connected layer (F6), that are followed by the output layer. Convolutional layers use 5 by 5 convolutions with stride 1.


**What were some problems with the initial architecture?**

With having so many data points in the training set , basic LeNet-5 Model was not able to train on the images properly.
since the model was generating so many feature maps , the model trained on training set so perfectly that performed poorly on validation set.

**How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.**

The Model Overfitting since the validation accuracy was too low , as compared to Training Accuracy. Previously the model was achieving 96% Training Accuracy and 91% validation accuracy.

To adjust this ,I tried a range of learning rates , and Dropout combinations . I tried different batch_size to obtain a cleanly trained model, but failed.

Solution:

- L2_Regularization: I used weight decay to tackle with the situation

 `variables = tf.trainable_variables()
  l2_regularizer = tf.add_n([ tf.nn.l2_loss(v) for v in variables if '_b' not in v.name ])
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = one_hot_y, logits= logits)   
  loss_operation = tf.reduce_sum(cross_entropy) + beta * l2_regularizer`

 _I skipped L2_Regularization of biases as they were small in numbers and they would almost affect neglibily in the training_

**Which parameters were tuned? How were they adjusted and why?**

 Learning_rate was adjusted from 0.00001 to 0.0005 : because the learning curve was converging very slowly
 BATCH_SIZE was adjusted from 32 to 128            : because the learning curve was converging very slowly.     
 EPOCHS were restricted to 20 instead of 40        : High Epochs were overfitting the model

**What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?**

While Designing the model few key considerations were :

- Convolution layer :  CNNs are the foundation of modern state-of-the art deep learning-based computer vision. For the model to have: local receptive fields, shared weights and spacial subsampling Convolution Layer is the best option.
Because of its shared weights there are less number of trainable parameters.

To solve this problem , we need to extract maximum information from an image , ex [lines , shape , grids , combinantions of these forming any part of the image etc.], in this method the trained model can pick features of the targeted image from any part of the image.

- Dropout Layer: Dropout Layer prevents the model from overfitting . It partially removes the neural links going forward , so that the model trains itself, in their absence.

- L2_regularization: Also known as weight decay , helps in reducing the overfitting .It helps the model to have neither high Bias not high Variance.


**How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?**

Higher Accuracy on Training Set doesn't provide enough evidence to say that it can successfully classify any given data ,
But Higher accuracy on Validation Data proves that we the neural network hyperparameters are tuned to perfection

The Test Accuracy of 94.8 %  shows that the model can now classify any new image with high success rate.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 10 German traffic signs that I found on the web:

![Slippery road][image16]        ![Roundabout mandatory][image25] ![Bumpy road][image17]
![Road work][image18]            ![Children crossing][image19]
![Speed limit (50km/h)][image20] ![Speed limit (30km/h)][image21]
![Yield][image22]  ![Go straight or right][image23]
![Turn right ahead][image24]  


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Slippery road         | Slippery road 							    |
| Roundabout mandatory	| Roundabout mandatory      					|
| Bumpy road            | No passing                                    |
| Road work             | Road work 								    |
| Children crossing     | Children crossing						        |
| Speed limit (50km/h)	| Roundabout mandatory				 			|
|Speed limit (30km/h)   | Yield                                         |
|Yield                  | Yield                                         |
|Go straight or right   | Go straight or right                          |
|Turn right ahead       | Turn right ahead                              |

Prediction Rate : 70%

The model was able to correctly guess 7 of the 10 traffic signs


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the [105th -111th] cell of the Ipython notebook.


![softmax_prob][image14]

The model performed fairly well on the new test set.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Convolution Layer 1:

![Conv_layer_1][image12]


Here we can see that for 32x32 grayscale image , for the very first layer the feature maps extracted all the edges of the given Image, and picked the soft , hard edged as well. It picked lines , shape from the image

Convolution Layer 2:

![Conv_Layer_2][image13]

Since the 1st convolution layer was followed by RELU and 2x2 maxpooling , in 2nd Convolution Layer the feature map extracted the pixel information which makeup the lines , shapes and edges in the 1st convolution Layer
