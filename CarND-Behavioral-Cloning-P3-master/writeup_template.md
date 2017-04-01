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

![Figure 1](https://github.com/Chinchilla1988/CarND-Behavioral-Cloning-P3-master/tree/master/CarND-Behavioral-Cloning-P3-master/examples/givendataset.png)

To balance the given Dataset I decided to cut the given Dataset to approximate a Gaussian distribution.

## 2. Data Cutting and Data Augmentation
---
First I separated all Indices with a steering angle equally zero and all Indices unequally zero. Then I created a Vector which contains all Indices with a steering angle unequally zero. Inside a for loop i searched to every index the corresponding images, flipped them and saved them afterwards in a separate vector. Same has been applied to all corresponding steering angles. That's the way i doubled the images to balance my Dataset. In the next step i added my vector containing the flipped images and steering angles and added them to the given dataset. 

![Figure 2](https://github.com/Chinchilla1988/CarND-Behavioral-Cloning-P3-master/tree/master/CarND-Behavioral-Cloning-P3-master/examples/augmented_Dataset.png)

As you can see the Dataset still contains a bias to a steering angle equally zero. In this step i downsampled all Indices regarding to a steering angle equally zero to a max sampleset of 100 Indices: 

bin=100
threshold=100

check=[]
index_cutplus=[]
index_cutminusl=[]
index_cutminus=[]
index_cutminusk=[]


for i in range(bin*2-2):
    k=-1.02+2*i/bin
    h=-1.+2*i/bin
    counter=0
    for j in range(len(index)):
        if (float(steer[index[j]])>=k and float(steer[index[j]])<=h):
            if counter<threshold:
                check.append(index[j])
                counter+=1
                if float(steer[index[j]])>= 0.35 and float(steer[index[j]])<=0.95:
                    l=index[j]
                    index_cutplus.append(l)
                if float(steer[index[j]])<=-0.35 and float(steer[index[j]])>=-.95:
                    t=index[j]
                    index_cutminus.append(t)
                
            else:
                continue

In image 3 you can observe the almost balanced dataset:

![Figure 3](https://github.com/Chinchilla1988/CarND-Behavioral-Cloning-P3-master/tree/master/CarND-Behavioral-Cloning-P3-master/examples/Almost_balanced.png)

The cutted Dataset is still not satisfying because it lacks several steering angles in the range of abs(alpha) >= 0.5. To raise the samples of all steering angles abs(alpha) >= 0.5 I upsampled all steering angles abs(alpha) >= 0.5 by the following function:

def createIndex(index,number_of_repetition):
    ind=[]
    for i in range(len(index)):
        for k in range(number_of_repetition):
            ind.append(index[i])
            
        
    return ind


In image 4 are the upsampled steering angles represented:

![Figure 4](https://github.com/Chinchilla1988/CarND-Behavioral-Cloning-P3-master/tree/master/CarND-Behavioral-Cloning-P3-master/examples/upsample.png?raw=true)

---
In the last step I added the upsampled indexvectors to the indexvector containing the indices from the given and the flipped dataset. Then the augmented Indexvector has been cutted. The final result is given in image5:
![Figure 5](https://github.com/Chinchilla1988/CarND-Behavioral-Cloning-P3-master/tree/master/CarND-Behavioral-Cloning-P3-master/examples/finalcut.png?raw=true)

---

## 3
## 3 Image Preprocessing
---

As stated in Chapter 1 I decided to train my model by Data Augmentation not by self driving. That's why I decided to use the following preprocess techniques:

The Function draw_image decides randomly which image will be drawn by a given Index to feed it to my pipeline and my model. By the randomly set number nb it decides if it's the left, center or right image. The drawn image is given a modified steering angle.

def draw_image(index,center, left, right,steer):
    nb=np.random.randint(3,size=1)

    if nb == 0:
        image=left[index]
        steering=float((steer[index])) + float((0.25))
        
    if nb == 1:
        image=center[index]
        steering=float(steer[index])
        
    if nb == 2:
        image=right[index]
        steering=float((steer[index])) - float((0.25))
        
    #plt.imshow(image)
    return image,steering


The Function brightness is scaling the V-Value of the transformed image. The image gets lighter or darker depending on the randomly drawn number.

def brightness(img,steering_angle):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    scale=0.2+np.random.uniform(0,1)
    img[:,:,2]=img[:,:,2]*scale
    img=cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    steer=steering_angle
    return img, steer

The function translate, translate the image horizontally and vertically and is manipulating the corresponding steering angle. The parameters have been tuned empirically.

def translate(img,steering_angle,tx,ty):
    measurement=None
    nb=None
    k=None
    
    rows,cols = img.shape[0],img.shape[1]
    nb=np.random.randint(2,size=1)
    k=np.random.randint(2,size=1)
    
    trans=tx*random.uniform(0,1)
    trans_up=ty*random.uniform(0,1)
    #print("nb: "+str(nb))
    
    if nb == 0:
        if k == 0:
            ## shift left
            
            M = np.float32([[1,0,-trans],[0,1,0]])
            
            measurement=float(steering_angle+(trans/cols)*1.5)
            
        if k == 1:
            ## shift right
            
            M = np.float32([[1,0,trans],[0,1,0]])
            
            measurement=float(steering_angle-(trans/cols)*1.5)
            
        #print('shift '+str(k))
        img = cv2.warpAffine(img,M,(cols,rows))

        
        ### lenkwinkeländerung eintragen
    ## vertical shift
    if nb == 1:
        if k == 0:
            ## shift up
            M = np.float32([[1,0,0],[0,1,trans_up]])
            
            measurement=float(steering_angle)
            
        if k == 1:
            ## shift down
            M = np.float32([[1,0,0],[0,1,-trans_up]])
            
            measurement=float(steering_angle)
            
        img = cv2.warpAffine(img,M,(cols,rows))    
    
    return img,measurement

The function rot is rotating the image. The steering angle manipulation factor has been tuned empirically
def rot(img,steering_angle):
    rows,cols = img.shape[0],img.shape[1]
    scale=np.random.uniform(0,1)
    angle_rot=25*scale
    k=np.random.randint(low=0,high=2,size=1)
    scale=np.random.uniform(0,1)
    if k == 0:
        ## rot left
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle_rot,1)
       #        measurement=float(steering_angle+(angle_rot*8/360))

        measurement=float(steering_angle+(angle_rot*10/360))
        
    if k == 1:
        ## rot right
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-angle_rot,1)
        #        measurement=float(steering_angle+(angle_rot*8/360))

        measurement=float(steering_angle-(angle_rot*10/360))
    
   
    img = cv2.warpAffine(img,M,(cols,rows))   
    return img, measurement
    
As the functionname states this function crops and resizes the image.

def crop_image(img,steering_angle):
    crop_img=img[60:140,0:360]
    res = cv2.resize(crop_img,(200, 66), interpolation = cv2.INTER_LINEAR)
    measurement=float(steering_angle)
    return res,measurement

---

## 4. Nvidia Model

I decided to implement the Nvidia Model because it was succesfully implemented to drive a real car.
The architecture is presented in image 6.

![Figure 6](https://github.com/Chinchilla1988/CarND-Behavioral-Cloning-P3-master/tree/master/CarND-Behavioral-Cloning-P3-master/examples/nvidia.png?raw=true)









# description in here






## 5. Data Generator

The function generator has been modified. It doesn't strip images as stated in the lecture. It receives as Input the given Index as a scalar; center, left and right images and the steering angle as a vector. The image and steering angle is chosen by the given index.

def generator(index,center,left,right,steering_angle, batch_size=32):
    num_samples=len(index)
    np.random.shuffle(index)
    counter=0
    
    
    while 1: # Loop forever so the generator never terminates
        #np.random.shuffle(index)
        
        
        
        for offset in range(0, num_samples, batch_size):
            images = []
            angles = []
            
            batch_samples = index[offset:offset+batch_size]
            

            
            for batch_sample in batch_samples:
                image,angle = draw_image(batch_sample,center,left, right,steering_angle)
                img,ang=pipe(image,angle)
                
                
                images.append(img)
                angles.append(float(ang))                
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)



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
