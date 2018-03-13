# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_lane.png "Center lane driving"
[image3]: ./examples/recovery1.jpg "Recovery Image"
[image4]: ./examples/recovery2.jpg "Recovery Image"
[image5]: ./examples/recovery3.jpg "Recovery Image"
[image6]: ./examples/ex_normal.jpg "Normal Image"
[image7]: ./examples/flipped.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 149-159) 

The model includes RELU layers to introduce nonlinearity (code line 150-154), and the data is normalized in the model using a Keras lambda layer (code line 71). 

#### 2. Attempts to reduce overfitting in the model



The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 168-170). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 165).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and extra data for sharp bend driving. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use convolutional network to extract features from the road pictures in the 3 camera videos and learn the steering Angle command.

My first step was to use a convolution neural network model similar to the Lenet. I thought this model might be appropriate because it gives a very good accuracy on Imagenet database.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I took the left and right camera images too and used as a fixed offset to the steering angle as if those were seen by the centrall camera.

Then I switched to a more advanced Model architecture used by Autonomous driving group in NVIDIA.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially at sharp bends. To improve the driving behavior in these cases, I ran the simulator in training mode around sharp bends and appened into training data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 149-159) consisted of a convolution neural network with the following layers and layer sizes:
1)First convolutional layer of 5x5 filter and filter depth of 24. Stride is 2 and Activation ReLu
2)Second convolutional layer of 5x5 filter and filter depth of 36. Stride is 2 and Activation ReLu
3)Third convolutional layer of 5X5 filter and filter depth of 48. Stride is 2 and Activation ReLu
4)Fourth convolutional layer of 3x3 filter and filter depth of 64. Stride is 1 and Activation ReLu
5)Fifth convolutional layer of 3x3 filter and filter depth of 64. Stride is 1 and Activation ReLu
6)Then a Flatten layer
7)Then a Fully connected layer of output 100 features
8)Then a Fully connected layer of output 50 features
9)Then a Fully connected layer of output 10 features
10)Then a Fully connected layer of output 1 feature



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it daviates from  road in real autonomous driving.These images show what a recovery looks like starting from outside of the road on right side to the middle of the road :

![alt text][image3]
![alt text][image4]
![alt text][image5]



To augment the data sat, I also flipped images and angles thinking that this would help the model to generalize better. For example, here is an image that has then been flipped:

![alt text][image6] to
![alt text][image7]

I also, took the left camera and right camera images. For left camera images I added a fixed offset of 0.2 to the steering angle as if center camera is seeing the image. For the write camera image, a fixed offset of 0.2 was subtracted from the steering angle.

After the collection process, I had 7962 data points. I then preprocessed this data by 
(1)centering the images around zero with small standard deviation
(2)cropping the top 50 pixels and bottom 20 pixels of the images.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 because after that the validation set loss was increasing, meaning the model was overfitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
