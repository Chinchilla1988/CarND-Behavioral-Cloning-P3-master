#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

*Table of Contents*

* 1. Data Visualisation
* 2. Data Cutting and Data Augmentation
* 3. Image preprocessing
* 4. Nvidia DNN-Model
* 5. Data Generator
* 6. Epoch and Loss


## 1. Data Visualisation
---
I decided to work with the given Udacity Dataset because I wanted to focus myself on the topic of Data Augmentation. My goal was to train my model so my car is going to pass both tracks only by using Data Augmentation.
In the first step i visualized the given dataset. As you can see in image 1 the given Dataset has a strong bias related to a steering angle equally zero.

[![Given Dataset](https://github.com/Chinchilla1988/CarND-Behavioral-Cloning-P3-master/tree/master/CarND-Behavioral-Cloning-P3-master/examples/givendataset.png)](https://github.com/Chinchilla1988/CarND-Behavioral-Cloning-P3-master/tree/master/CarND-Behavioral-Cloning-P3-master/examples/givendataset.png)

To balance the given Dataset I decided to cut the given Dataset to approximate a Gaussian distribution.

## 2. Data Cutting and Data Augmentation
---
First I separated all Indices with a steering angle equally zero and all Indices unequally zero. Then I created a Vector which contains all Indices with a steering angle unequally zero. Inside a for loop i searched to every index the corresponding images, flipped them and saved them afterwards in a separate vector. Same has been applied to all corresponding steering angles. That's the way i doubled the images to balance my Dataset. In the next step i added my vector containing the flipped images and steering angles and added them to the given dataset. 

![Figure 2](https://github.com/Chinchilla1988/CarND-Behavioral-Cloning-P3-master/tree/master/CarND-Behavioral-Cloning-P3-master/examples/augmented_Dataset.png)

As you can see the Dataset still contains a bias to a steering angle equally zero. In this step i downsampled all Indices regarding to a steering angle equally zero to a max sampleset of 100 Indices.
In image 3 you can observe the almost balanced dataset:

![Figure 3](https://github.com/Chinchilla1988/CarND-Behavioral-Cloning-P3-master/tree/master/CarND-Behavioral-Cloning-P3-master/examples/Almost_balanced.png)

The cutted Dataset is still not satisfying because it lacks several steering angles in the range of abs(alpha) >= 0.5. To raise the samples of all steering angles abs(alpha) >= 0.5 I upsampled all steering angles abs(alpha) >= 0.5.

In image 4 are the upsampled steering angles represented:

![Figure 4](https://github.com/Chinchilla1988/CarND-Behavioral-Cloning-P3-master/tree/master/CarND-Behavioral-Cloning-P3-master/examples/upsample.png?raw=true)

---
In the last step I added the upsampled indexvectors to the indexvector containing the indices from the given and the flipped dataset. Then the augmented Indexvector has been cutted. The final result is given in image5:
![Figure 5](https://github.com/Chinchilla1988/CarND-Behavioral-Cloning-P3-master/tree/master/CarND-Behavioral-Cloning-P3-master/examples/finalcut.png?raw=true)

---
## 3 Image Preprocessing
As stated in Chapter 1 I decided to train my model by Data Augmentation not by self driving. That's why I decided to use the following preprocess techniques:
The Function draw_image decides randomly which image will be drawn by a given Index to feed it to my pipeline and my model. By the randomly set number nb it decides if it's the left, center or right image. The drawn image is given a modified steering angle.
The Function brightness is scaling the V-Value of the transformed image. The image gets lighter or darker depending on the randomly drawn number.
The function translate, translate the image horizontally and vertically and is manipulating the corresponding steering angle. The parameters have been tuned empirically.
The function rot is rotating the image. The steering angle manipulation factor has been tuned empirically  
As the functionname states this function crops and resizes the image.

## 4. Nvidia Model

I decided to implement the Nvidia Model because it was succesfully implemented to drive a real car.
The architecture is presented in image 6.

![Figure 6](https://github.com/Chinchilla1988/CarND-Behavioral-Cloning-P3-master/tree/master/CarND-Behavioral-Cloning-P3-master/examples/nvidia.png?raw=true)









# description in here

## 5. Data Generator

The function generator has been modified. It doesn't strip images as stated in the lecture. It receives as Input the given Index as a scalar; center, left and right images and the steering angle as a vector. The image and steering angle is chosen by the given index.

## 6 Epoch and Loss

Because of the heavy Data augmentation my DNN converge slowly. I had to train my model for 125 Epochs. 

Epoch 1: 
You're running in loop: 1
Epoch 1/5
9488/9488 [==============================] - 92s - loss: 0.4778 - val_loss: 0.2366

Epoch 125:
Epoch 5/5
9488/9488 [==============================] - 55s - loss: 0.1872 - val_loss: 0.1738
## 7 Lessons learned

Data Augmentation is a powerful technique to create your own unique dataset. Also I don't have to waste my time driving by myself to create the necessary data. 
My model drives track 1 fully autonomously.

On Track 2 my model drives mostly autonomous. It fails to detect sharp curves. The rest of the track it drives fully autonomously through track 2.
Also my model stops converging at a training loss of 18.72 %. I guess it's stuck in a local minima and if I solve this problem my model drives also through Track 2 by itself. If i have some time left in the upcoming weeks I'll tune my model but for now I have to stop. 

Because the DNN is not deterministic it's very important to feed the DNN with good data. Without good data the model will fail. Also it's very important to choose or create an architecture which can detect all necessary Data to predict a proper steering angle.
