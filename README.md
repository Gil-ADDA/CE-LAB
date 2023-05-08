

## CE-LAB  Image Classification Model Using Edge Impulse







# Project Overview

This project was inspired by the need for better engagement with the devices and projects in the Connected Environment Lab, both for students and guests visiting the lab. The primary objective is to detect two unique objects in the lab - the wind spiral and wind barometer - using an image classification program. By successfully identifying these objects, users can better interact with and learn about the lab's devices.

The project employs a transfer learning approach to train an image classification model for detecting the wind spiral and wind barometer objects. A dataset consisting of 149 images, representing the two object labels, was used for training the model. Transfer learning leverages a pre-trained model to fine-tune the image classification task, helping to achieve better performance despite the smaller dataset.

The image classification model can be accessed by students and guests in the lab to help them identify the wind spiral and wind barometer objects. This interactive experience aims to foster better engagement and understanding of the devices in the lab, benefiting both students and guests. Possible future work includes improving the model's performance by collecting a more diverse dataset and incorporating an object detection model.


# Background 
The project was motivated by the need to improve engagement with the devices and projects in the Connected Environment Lab, both for students and guests. The primary objective is to detect two unique objects in the lab - the wind spiral and wind barometer - using an image classification program. The lab receives a lot of visitors, and with the increasing number of devices and projects, it can be challenging for visitors to understand the narrative and functionality of each project. The wind spiral and wind barometer are two examples of devices that visitors find intriguing but may not understand fully without the help of a teaching staff member. The project's short-term benefits include allowing guests to interact with and learn about these devices in a more meaningful way. The long-term goal is to create a virtual hub of all the projects in the lab, which will be open to everyone. Students and guests can connect with the co-creators of the projects and consult with them, potentially leading to the evolution of these projects or the creation of similar ones. The dataset used for this project consists of 149 images for both wind spiral and wind barometer classes. While working on the project, I faced challenges in achieving good results and figuring out the right amount of data required for the project to work. After experimenting with different approaches, I decided to pivot to transfer learning based on recommendations from the book Algorithms for Edge AI. 

![Object detection bounding box attempts](https://github.com/Gil-ADDA/CE-LAB/blob/d6dc0140d765485b9b29957252cfe8582ada2612/image/Object%20detection%20bounding%20box%20attempts.png)



# Getting Started Image Classification Model Using Transfer Learning
## 1. Dataset Collection and Split
I started by collecting a dataset of 149 images, consisting of two labels representing the unique objects in the lab - the wind spiral and wind barometer. I divided the dataset into training and test sets, following a 77% training and 23% test split, which is close to the commonly recommended 80% training and 20% test split.
![split of the data](https://github.com/Gil-ADDA/CE-LAB/blob/5dd489074ee6368cd160508a72b5791f405d4a10/image/split%20of%20the%20data.png)

## 2. Image Upload and Preprocessing block Configuration 
Next, I uploaded the images to the Edge Impulse platform. The images were resized to 96x96 pixels, and preprocessing and normalization were applied to the image data. The color depth was kept as RGB. 
I set up a processing block for the image data, which preprocesses and normalizes the images.
I configured a learning block using transfer learning for images. This fine-tunes a pre-trained image classification model on my dataset.


## 3. Data Exploration and Feature Visualization
Throughout the process, Edge Impulse's Data Explorer tool was used to identify the characteristics of each image and how the model categorized them. Data Explorer shows all project data and was created by passing input through a pre-trained visual model. This allows for fast analysis of the data, including feature distributions and relations between each of the images. Pre-trained model data exploration has advantages like faster results and improved performance compared to training from scratch. When data sets are small, this pre-trained model knowledge can be adjusted to suit the data.

![Data Explorer](https://github.com/Gil-ADDA/CE-LAB/blob/4bc06f64ee0e758ea9d4a4ba2524ebef6b4640e2/image/Data%20Explorer.png)

## 4. Model Selection
For the image classification task, I chose to use transfer learning with the MobileNetV2 96x96 1.0 model (no final dense layer, 0.3 dropout) to leverage the knowledge of a pre-trained model and fine-tune it on my dataset. This approach allows for improved performance even with relatively small image datasets.


## 5. Model Training
I trained the model using the processed images and learning block configuration with a batch size of 32 and 40 epochs. To address overfitting issues, I modified the Keras expert mode settings, setting the learning rate to 0.0015. I didn't use the auto-balance dataset or data augmentation options in this case, as I was not aware of the need to add them manually in the Keras code at the time. Despite this, the model testing results showed an accuracy of 94.29%, indicating that the model is effective in identifying the wind spiral and wind barometer objects. However, it's important to consider using data augmentation and class balancing techniques in future projects, especially when working with a small dataset like mine. Data augmentation can help prevent overfitting by artificially increasing the size of the training dataset and adding diversity to the data. Class balancing can ensure that the model has equal representation of all classes, even when one class has significantly more samples than another.

![Transfer Learning Documentation](https://github.com/Gil-ADDA/CE-LAB/blob/78db88dd2031858951dd2e66a405a8f9987847d3/image/transfer%20learning%20documention%20.png)

![Training Data](https://github.com/Gil-ADDA/CE-LAB/blob/4bc06f64ee0e758ea9d4a4ba2524ebef6b4640e2/image/Training%20Data.png)




## 6. Model Testing and Result
After training the image classification model using transfer learning, I tested the model's performance using the 23% of the dataset that was set aside for testing. I classified all the test images using the model and set the threshold at 0.8 (80%). The model testing results showed an accuracy of 93.5%, indicating that the model is effective in identifying the wind spiral and wind barometer objects.

![Result of the Model](https://github.com/Gil-ADDA/CE-LAB/blob/fe0ba971f1ac7ddcd916a45734f81571d11a0c33/image/Result%20of%20the%20model1.png)

![Test Data 88%](https://github.com/Gil-ADDA/CE-LAB/blob/4bc06f64ee0e758ea9d4a4ba2524ebef6b4640e2/image/Test%20Data%2088%25.png)

![QR Code](https://github.com/Gil-ADDA/CE-LAB/blob/78db88dd2031858951dd2e66a405a8f9987847d3/image/QR-CODE.png)








### Data Collection and Preprocessing

The process of collecting images of the two unique objects in the CE-LAB involved taking photographs of the wind spiral and wind barometer using a smartphone camera. The images were then transferred to a computer for preprocessing, 

### Model Selection and Training

Discuss why you chose image classification over object detection, and describe the model architecture, training process, and any challenges faced.

### Deployment and Usage

Explain how to deploy the model on a phone using Edge Impulse, and provide instructions for using the app for object detection.





# Video Preview of the Model 
[![Video Thumbnail](https://img.youtube.com/vi/nQ7Ruwu12t8/0.jpg)](https://www.youtube.com/watch?v=nQ7Ruwu12t8)

# Challenges faced during development
Obtaining accurate results proved difficult during development. Additionally, finding the right dataset size was an issue; only 20 images were initially used and more were added as time went on, including images without the object and changes to the background. To gain a better understanding of machine learning, I employed various techniques and asked the Edge Impulse community for guidance (Attached underneath this paragraph). I studied other projects, articles, and books for additional insights. I also tweaked the model's variables, like architecture and image size, to boost performance.
Moreover before considering the question of how much data is required for your personal project, it is suggested to read the chapter on "Estimating Data Requirements" in the book "AI at the Edge" by Jason M. Pittman (2021) 
![Edge impusle community](https://github.com/Gil-ADDA/CE-LAB/blob/af24f8076a5d3730215290e521f82f2a187943f6/image/Edge%20impulse%20community.png)

The Link to the forum [https://forum.edgeimpulse.com/t/improve-performance-of-an-object-detection/6922/7](https://forum.edgeimpulse.com/t/improve-performance-of-an-object-detection/6922/7)



# Critical Reflection and Learning

Reflect on your experiences, any challenges faced, and potential improvements to the project.

![Data requirmentes Table](https://github.com/Gil-ADDA/CE-LAB/blob/1591c633fefef7717f39c174b17560ccd6428153/image/Data%20requirements%20for%20common%20tasks.png)

# Workspace
Edge Impulse 
<br> <img src="https://www.edge-ai-vision.com/wp-content/uploads/2021/05/logo_edgeimpulse_may_2021.png" width="80" height="60">

Google Sheets <br> <img src="https://www.gstatic.com/images/branding/product/1x/sheets_2020q4_32dp.png" width="60" height="60"> (Used for documenting the results of training and testing)

### The link to the original Sheet (Object detection and transfer learning)  https://docs.google.com/spreadsheets/d/1LEAihPLvVjIKXl6ZXBGP7pdwTo1F_FSEUE8_gQM_ccs/edit?usp=sharing
## Contact Details
[<img src="https://img.icons8.com/color/48/000000/gmail.png"/>](mailto:giloo1047@gmail.com)
[<img src="https://img.icons8.com/color/48/000000/linkedin.png"/>](https://www.linkedin.com/in/gil-adda-16385510b/)



# Future work


## Refrences
Edge Impulse. (n.d.). Object detection. Retrieved March 18, 2023, from https://docs.edgeimpulse.com/docs/tutorials/object-detection

O'Reilly Media, Inc. (2023). AI at the Edge: A Developer's Guide. O'Reilly Media, Inc. Retrieved from https://learning.oreilly.com/library/view/ai-at-the/9781098120191/


## MIT License

Copyright (c) 2023 Gil Adda

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

