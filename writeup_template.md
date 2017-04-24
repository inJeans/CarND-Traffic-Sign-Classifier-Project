#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./writeup-images/dataset.png "Traffic sign dataset visualisation"
[image2]: ./writeup-images/preprocess.png "Preprocessed image coverting to YUV and using GCN"
[image3]: ./writeup-images/augmented.png "Augmented images"
[image4]: ./test-images/4-speed_limit_70.png "Traffic Sign 1"
[image5]: ./test-images/11-right_of_way.jpg "Traffic Sign 2"
[image6]: ./test-images/14-stop.jpg "Traffic Sign 3"
[image7]: ./test-images/18-general_caution.jpg "Traffic Sign 4"
[image8]: ./test-images/22-bumpy_road.jpg "Traffic Sign 5"
[image9]: ./test-images/25-road_work.jpg "Traffic Sign 6"
[image10]: ./test-images/27-pedestrians.jpg "Traffic Sign 7"
[image11]: ./writeup-images/bumpy.png "Preprocessed bumpy image"
[image12]: ./writeup-images/70kmh.png "70 km/h"
[image13]: ./writeup-images/pedestrian.png "Preprocessed pedestrian image"
[image14]: ./writeup-images/conv1.png "Activations for convolution 1"
[image15]: ./writeup-images/conv2.png "Activations for convolution 2"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/inJeans/CarND-Traffic-Sign-Classifier-Project)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Using simple numpy array properties and Python's built in len function you can see:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data is distributed across each class.

![alt text][image1]

You can see that the training set is distributed very uniformly, so our trained network may be biased towards certain types of signs. The 'Gerneral Caution' sign was the least represented while we had the most examples of the 'Speed limit (30km/h)'

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Following the preprocessing steps taken in the provided baseline paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks," the images were first converted to the YUV colourspace using the scikit image library (this function will return the luminance values in the range [0,1]). I then applied global contrast normalisation to each image using the description given in Goodfellow et al's book "Deep Learning" (Chapter 12, section 12.2.1.1, Contrast Normalization). I did not implement the local contrast normalisation for this project. 

I also through in a little Gaussian noise for good measure. Given our dataset is so small I thought I might try to build in some regularisation. As I will discuss later on, my goal for this particular project was to see how far I could push the DL model with only image processing techniques, rather than by developing a large and complex DL architecture. So rather than use dropout or some other regularisation technique, I am instead adding it here in the preprocessing stage. 

Here is an example of a traffic sign image before and after preprocessing

![alt text][image2]

I then also added the data augmentation steps used in the paper which where rotation and scaling. Rotations were (as in the paper) lrandomly chosen between +-10 degree rotations about the center, and scaling was randomly chosen between a scaling factor of 0.1 and 2. Both of these augmentations could easily be performed using the `transform.rotate` and `transform.rescale` functions from the scikitimage library. Each image had 5 random augmentations performed.

Here is an example of an original image and its augmented images:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I have kept the standard LeNet model we developed in earlier classes (except I increaed the number of output in the final two layers). I really think that there is an over emphasise on developing ever more complicated models and I wanted to see if I could acheive an acceptable result with minimal (to no) changes to the simplest convnet we have implemented. Below is the final architecture I went with:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x1		|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6    				|
| Fully connected		| outputs 256       							|
| RELU					|												|
| Fully connected       | outputs 128                                   |
| RELU					|												|
| Softmax				| output 43    									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Again keeping it simple. I used the adam optimiser, leaving the default values for optimisation. I started with a pretty standard value of 0.001 for the learning rate and evolved the model for 20 epochs. This seemed to produce fairly reasonable results.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.85%
* validation set accuracy of 94.1% 
* test set accuracy of 92.9%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I chose the LeNet architecture we had developed in previous classes. I was trying to acheive the best result I could with the simplest model possible

* What were some problems with the initial architecture?

Initially I my preprocessed images did not include any artificial noise and the standard LeNet we had been using performed quite well (validation accuracies around 93%). However once I added in the artificial noise I had to increase the learning capacity of the network by adding a few extra outputs to the final two layers. I also reduced the learning rate down to 0.001 (initially it was 0.002) and trained for a few extra epochs, twenty in total.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

As I mentioned above I increased the number of outputs in the final two dense layers to increase the learning capacity of the network. This seeed to allow it to handle noisy images a lot better.

* Which parameters were tuned? How were they adjusted and why?

I tuned the learning rate and number of epochs via a very dumb, human-in-the-loop trial and error method which involved only a few iterations. This seemed enough to acheive the required accuracy, there is no need to over engineer the solution here.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

As I mentioned, my main design choice was to create the simplest model possible. Given that we have such a small dataset, it would be quite easy to overfit if we used one of the larger more popular network architectures. I started from the LeNet architecture because it has been shown to work quite well for simple image classification problems (originally black and white number images). Given that our traffic signs have no extra information across the different colour channels (colour is mainly used for attracting divers' attetions), we could assume that black and white images would suffice. However, I did, in the end, extend the network into a three channel colour space to match the preprocessing steps made in the benchmark reference material.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10]

The first image might be difficult to classify because it is not tightly cropped to the sign, likewise the pedestrian signa and bumpy road sign are a bit far away. There is a lot of background and so when we downsample the image we might lose a lot of the important information.

The road work image has sme other distracting red and white stripes, so these may also confuse the network given that all of the signs are rimmed with a red border.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                  |     Prediction	        					| 
|:-------------------------------:|:-------------------------------------------:| 
| 70 km/h      		              | Children crossing							| 
| Right of way at next interstion | Right of way at next intersection			|
| Stop					          | Stop										|
| General caution	      		  | General caution 			 				|
| Bumpy road			          | Go straight or left							|
| Road work                       | Right of way at next intersection           |
| Pedestrains                     | Road work                                   |


The model was able to correctly guess 3 of the 7 traffic signs, which gives an accuracy of 42%. This compares poorly to the accuracy on the test set of 92%. However as discussed some of these images were quite poor once downsampled to the 32x32 input size. In particular the bumpy road, and 70 km/h signs were completely indistinguishable.

![alt text][image11] ![alt text][image12]

I guess the pedestrian kind of looks like the road worker in the sign so that confusion is also acceptable.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The mode is very confident with all of its predicitions. All of the top-1 accuracies are above 99%!

70 km/h:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Children crossing								| 
| .000     				| Bicycles crossing								|
| .000					| 30 km/h   									|
| .000	      			| 20 km/h   					 				|
| .000				    | Slippery Road      							|


Road work:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Right of way at next intersection				| 
| .000     				| 20 km/h       								|
| .000					| 30 km/h   									|
| .000	      			| 50 km/h   					 				|
| .000				    | 60 km/h           							|

Bumpy road:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.000         			| Go straight or left							| 
| .000     				| No passing for vehicles over 3.5 tonnes		|
| .000					| Slippery road									|
| .000	      			| Priority road					 				|
| .000				    | End of no passing by vehicles over 3.5 tonnes	|

Pedestrians:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Road work     								| 
| .001     				| 20 km/h					         			|
| .000					| 60 km/h   									|
| .000	      			| End of speed limit 80 km/h	 				|
| .000				    | 50 km/h           							|

General Caution:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.000         			| General caution								| 
| .000     				| 20 km/h								        |
| .000					| 30 km/h   									|
| .000	      			| 50 km/h   					 				|
| .000				    | 60 km/h            							|

Right of way at next intersection:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.000         			| Right of way at next intersection				| 
| .000     				| Beware of ice/snow    						|
| .000					| Road work  									|
| .000	      			| Road narrows on right			 				|
| .000				    | Slippery road        							|

Stop:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.000         			| Stop				            				| 
| .000     				| No vehicles        							|
| .000					| Road work   									|
| .000	      			| 30 km/h   					 				|
| .000				    | 50 km/h           							| 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image13]

It looks like the first convolution layer picks up the edges of the circle quite well. It also seems activated by the red and white parts of the sign. There does appear to be quite a high activation around the edge of the pedestrian figure in feature moaps 1 and 2, which is interesting given that this image was miss classified.

![alt text][image14]

It is quite hard to intuit any many from the second convolution filter directly. There doesn't appear to be any clear structure in any of the images.

![alt text][image15]
