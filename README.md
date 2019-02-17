# Behaviorial Cloning Project

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_lane.png "Center lane driving"
[image3]: ./examples/recovery1.jpg "Recovery Image"
[image4]: ./examples/recovery2.jpg "Recovery Image"
[image5]: ./examples/recovery3.jpg "Recovery Image"
[image6]: ./examples/ex_normal.jpg "Normal Image"
[image7]: ./examples/flipped.png "Flipped Image"


Overview
---
In this project, a convolutional neural network has been trained to clone human driving behavior. The model 'see's the environment through a front facing camera and generates steering angle output to an autonomous vehicle. The model is trained, validated and tested using Keras.

First, a car is steered around a track on udacity simulator for data collection. The collected image data and the steering angles is used to train a neural network and then this model is used to drive the car autonomously around the track.


The Project
---
The goals / steps of this project are the following:
* To collect data of good driving behavior using the simulator
* To design, train and validate a Keras model(convolutional Neutral Network) that predicts a steering angle from image data
* To use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.


### Dependencies

The following resources can be found in this github repository:
* drive.py
* video.py

The simulator was downloaded from the udacity classroom.

For deatils about seeting up the environment, see the udacity repo in mentioned in the Ref section below.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires the trained model to be saved as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.



### Note
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

## Used Model Architecture and Training

### model.py

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline used for training and validating the model, and it contains comments to explain how the code works.

### model architecture

The model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 149-159) 

The model includes RELU layers to introduce nonlinearity (code line 150-154), and the data is normalized in the model using a Keras lambda layer (code line 71). 

### Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 168-170). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 165).

### Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and extra data for sharp bend driving.

## Model Architecture and Training Strategy

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




## Reference
[Udacity Repo](https://github.com/udacity/CarND-Behavioral-Cloning-P3)

