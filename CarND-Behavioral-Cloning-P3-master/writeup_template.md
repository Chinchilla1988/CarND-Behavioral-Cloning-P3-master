#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

*Table of Contents*

 1. Data Visualisation
 2. Data Cutting and Data Augmentation
 3. Image preprocessing
 4. Nvidia DNN-Model
 5. Data Generator
 6. Epoch and Loss


## 1. Data Visualisation
---
I decided to work with the given Udacity Dataset because I wanted to focus myself on the topic of Data Augmentation. My goal was to train my model so my car is going to pass both tracks only by using Data Augmentation.
In the first step I visualized the given dataset. As you can see in image 1 the given Dataset has a strong bias related to a steering angle equally zero.

![Given Dataset](examples/givendataset.png?raw=true)

To balance the given Dataset I decided to cut the given Dataset.

## 2. Data Cutting and Data Augmentation
---
First I created a vector which contains all indices with a steering angle unequally zero. Inside a for loop I searched to every index the corresponding images, flipped them and saved them afterwards in a separate vector. Same has been applied to all corresponding steering angles. That's the way I doubled the images to balance my dataset. In the next step I added my vector containing the flipped images and steering angles and added them to the given dataset. 

![Figure 2](examples/augmented_Dataset.png?raw=true)

As you can see the Dataset still contains a bias to a steering angle equally zero. In this step I downsampled all Indices regarding to a steering angle equally zero to a max sampleset of 100 Indices.
In image 3 you can observe the almost balanced dataset:


![Figure 3](examples/Almost_balanced.png?raw=true)

The cutted dataset is still not satisfying because it lacks several steering angles in the range of abs(alpha) >= 0.5. For balancing all steering angles abs(alpha) >= 0.5 has been upsampled, as presented in image 4.

![Figure 4](examples/upsample.png?raw=true)

In the last step I added the all indices of the flipped images to the indexvector. The augmented indexvector contains all indices to all given and flipped images. In the final step the indexvector has been cut. The final result is given in image5:
![Figure 5](examples/finalcut.png?raw=true)

---
## 3 Image Preprocessing
As stated in Chapter 1 I decided to train my model by Data Augmentation not by self driving. That's why I decided to use the following preprocess techniques:

The function draw_image decides randomly which image will be drawn by a given Index. All steering angles are adjusted according to their view to teach the car to recover from wandering off the track.

Shadowaugmentation has been applied to teach the car how to behave if shadow appears on the track. Thanks to Vivek Yadav and his great Idea how to cast shadows on Images!

![Figure 6](examples/shadow.PNG?raw=true)


Brightness augmentation has been applied to an image randomly. To achieve this effect we transform the image to the HSV-Space and scale the V-Value. Afterwards we transformate the image back to the RGB-Space. It's used to teach the car that the road surface' color varies. Track 1 is darker than the surface of track 2:

![Figure 7](examples/bright.PNG?raw=true)


By adding a randomly  horizontally translation to the image we teach the car how to recover from wandering off the track. By adding a modified steering angle it's able to find back to the road. The vertical translation teaches the car how to deal with steep roads. if we add a positive vertical translation we teach the car how to behave while driving down a road and vice versa. By a vertical translational shift the steering angle will not be modified.

![Figure 8](examples/translate.PNG?raw=true)

Also adding rotation allows us to create images from curves which upsamples all images containing curves in the dataset. The steering angle is modified.

![Figure 9](examples/rot.PNG?raw=true)

In the last step we crop the image to cut out the Hood and the background of the image and resize it to 66*200. All preprocessing and data augmentation techniques are presented in the following image:

![Figure 10](examples/pipe.PNG?raw=true)

## 4. Nvidia Model

I decided to implement the Nvidia Model because it was succesfully implemented to drive a real car.
The architecture is presented in image 6.

![Figure 11](examples/nvidia.png?raw=true)









# description in here

## 5. Data Generator

The function generator has been modified. It doesn't strip images as stated in the lecture. It receives as Input the given Index as a scalar; center, left and right images and the steering angle as a vector. The image and steering angle is chosen by the given index.
![Figure 12](examples/generator.PNG?raw=true)

## 6 Epoch and Loss

I had to train my model for 125 Epochs to converge my model aginst a solution. 

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
