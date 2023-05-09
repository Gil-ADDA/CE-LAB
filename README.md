
## CE-LAB  Image Classification Model Using Edge Impulse

# Project Overview

This project was inspired by the need for better engagement with the devices and projects in the Connected Environment Lab, both for students and guests visiting the lab. The primary objective is to detect two unique objects in the lab - the wind spiral and wind barometer - using an image classification program. By successfully identifying these objects, users can better interact with and learn about the lab's devices.

The project employs a transfer learning approach to train an image classification model for detecting the wind spiral and wind barometer objects. A dataset consisting of 149 images, representing the two object labels, was used for training the model. Transfer learning leverages a pre-trained model to fine-tune the image classification task, helping to achieve better performance despite the smaller dataset.

The image classification model can be accessed by students and guests in the lab to help them identify the wind spiral and wind barometer objects. This interactive experience aims to foster better engagement and understanding of the devices in the lab, benefiting both students and guests. Possible future work includes improving the model's performance by collecting a more diverse dataset and incorporating an object detection model.

# problem statement 
The Connected Environment Lab has a variety of unique devices and projects that can be difficult for students and visitors to engage with and understand. In particular, the wind spiral and wind barometer objects are visually impressive but require explanation to fully appreciate. The goal of this project is to develop an image classification program using transfer learning that can accurately detect and identify these objects in order to enhance engagement and understanding of the devices in the lab. The challenge is to achieve high accuracy with a relatively small dataset

![WindS Objects](https://github.com/Gil-ADDA/CE-LAB/blob/7ab792b29093a8fc4cc896c16054d685e39c08ff/image/WindS%20Objects.png)
# Research question 

The research question for this project was initially formulated as "What is the impact of using an image classification program on increasing engagement with the devices in the Connected Environment Lab?" However, based on feedback from the teaching staff, the question was revised to be more focused on the requirements of the course. The new research question is: 

How can transfer learning be utilized to enhance the accuracy of classifying wind spiral and wind barometer using computer vision? (The two unique objects in the Connected Environment Lab) 

# Background 
The project was motivated by the need to improve engagement with the devices and projects in the Connected Environment Lab, both for students and guests. The primary objective is to detect two unique objects in the lab - the wind spiral and wind barometer - using an image classification program. The lab receives a lot of visitors, and with the increasing number of devices and projects, it can be challenging for visitors to understand the narrative and functionality of each project. The wind spiral and wind barometer are two examples of devices that visitors find intriguing but may not understand fully without the help of a teaching staff member. The project's short-term benefits include allowing guests to interact with and learn about these devices in a more meaningful way. The long-term goal is to create a virtual hub of all the projects in the lab, which will be open to everyone. Students and guests can connect with the co-creators of the projects and consult with them, potentially leading to the evolution of these projects or the creation of similar ones. The dataset used for this project consists of 149 images for both wind spiral and wind barometer classes. While working on the project, I faced challenges in achieving good results and figuring out the right amount of data required for the project to work. After experimenting with different approaches, I decided to pivot to transfer learning based on recommendations from the book Algorithms for Edge AI. 

![Object detection bounding box attempts](https://github.com/Gil-ADDA/CE-LAB/blob/d6dc0140d765485b9b29957252cfe8582ada2612/image/Object%20detection%20bounding%20box%20attempts.png)

# Getting Started Image Classification Model Using Transfer Learning
## 1. Dataset Collection and Split
To begin, I collected a dataset of 149 images, including two labels representing the unique objects in the lab - the wind spiral and wind barometer. I then divided the dataset into training and test sets, following a 77% training and 23% test split, which aligns with the commonly recommended 80% training and 20% test split. In the training set, there were 74 images of wind spirals and 40 images of wind barometers, while the test set had 11 images of wind barometers and 24 images of wind spirals.

Based on feedback from teaching staff, it was recommended that an equal number of images be included for each object in both training and test datasets. This would help ensure the model is trained and tested on each object equally and improve overall accuracy.  To provide optimal training and testing of the model, having an equal number of images for each object in the training and test datasets. This advice is supported by the "The Ideal Dataset" section of the book "AI at the Edge," which emphasizes the importance of having balanced datasets to avoid bias in the model(O'Reilly Media, Inc., 2023).

![split of the data](https://github.com/Gil-ADDA/CE-LAB/blob/5dd489074ee6368cd160508a72b5791f405d4a10/image/split%20of%20the%20data.png)

## 2. Image Upload and Preprocessing block Configuration 
Next, I uploaded the images to the Edge Impulse platform. The images were resized to 96x96 pixels, and preprocessing and normalization were applied to the image data. The color depth was kept as RGB. 
I set up a processing block for the image data, which preprocesses and normalizes the images.
I configured a learning block using transfer learning for images. This fine-tunes a pre-trained image classification model on my dataset.


## 3. Data Exploration and Feature Visualization
Throughout the process, Edge Impulse's Data Explorer tool was used to identify the characteristics of each image and how the model categorized them. Data Explorer shows all project data and was created by passing input through a pre-trained visual model. This allows for fast analysis of the data, including feature distributions and relations between each of the images. Pre-trained model data exploration has advantages like faster results and improved performance compared to training from scratch. When data sets are small, this pre-trained model knowledge can be adjusted to suit the data.

![Data Explorerr](https://github.com/Gil-ADDA/CE-LAB/blob/70664a08f860f2fc9c783fa6262c89780430bca4/image/Data%20Explorerr.png)

## 4. Model Selection
To improve the performance of the image classification task, I employed transfer learning with the MobileNetV2 96x96 1.0 model, which was pre-trained on a large dataset and fine-tuned on my small dataset. This approach proved effective in achieving better results even with a limited amount of data. Additionally, the comparison of the experimentation results of the transfer learning process can be found in the Model Training section of this project's repository.

Moreover, based on the Model Architectures section in Chapter 4 of the book "AI at the Edge" by O'Reilly Media, I would recommend sticking with the MobileNetV2 architecture as it is suitable for deployment on mobile devices. The pre-trained MobileNetV2 model provides an excellent starting point for transfer learning and can be fine-tuned to achieve better performance on specific tasks.

## 5. Model Training
I trained the model using the processed images and learning block configuration with a batch size of 32 and 40 epochs. To address overfitting issues, I modified the Keras expert mode settings, setting the learning rate to 0.0015. I didn't use the auto-balance dataset or data augmentation options in this case, as I was not aware of the need to add them manually in the Keras code at the time. Despite this, the model testing results showed an accuracy of 94.29%, indicating that the model is effective in identifying the wind spiral and wind barometer objects. However, it's important to consider using data augmentation and class balancing techniques in future projects, especially when working with a small dataset like mine. Data augmentation can help prevent overfitting by artificially increasing the size of the training dataset and adding diversity to the data. Class balancing can ensure that the model has equal representation of all classes, even when one class has significantly more samples than another.

![Transfer Learning Documentation](https://github.com/Gil-ADDA/CE-LAB/blob/78db88dd2031858951dd2e66a405a8f9987847d3/image/transfer%20learning%20documention%20.png)

![Training Data](https://github.com/Gil-ADDA/CE-LAB/blob/4bc06f64ee0e758ea9d4a4ba2524ebef6b4640e2/image/Training%20Data.png)


## 6. Model Testing and Result

After training the image classification model with transfer learning, I tested it with 23% of the dataset and achieved an accuracy of 93.5%. To determine the optimal validation set size, I experimented with various parameters and found good performance with a size of 40. I classified all the test images using the model and set the threshold at 0.8 (80%). In March 2022, EON Tuner was only available for enterprise use, which may be useful for identifying appropriate architecture for ML projects.

It's worth noting that while this model performed well, there is always a risk of overfitting when training on a small dataset. To address this, I plan to further validate the model on additional datasets in the future.
![Result of the Model](https://github.com/Gil-ADDA/CE-LAB/blob/fe0ba971f1ac7ddcd916a45734f81571d11a0c33/image/Result%20of%20the%20model1.png)

![Test Data 88%](https://github.com/Gil-ADDA/CE-LAB/blob/4bc06f64ee0e758ea9d4a4ba2524ebef6b4640e2/image/Test%20Data%2088%25.png)

![QR Code](https://github.com/Gil-ADDA/CE-LAB/blob/41c259b2ea9fcb6b82219b435a827d8d70bb0494/image/QR-CODE%20Small.jpeg)

# Optimizing Image Dataset for Effective Model Training in Machine Learning Project

Upon starting my machine learning project, my goal was to train a model in Edge Impulse to distinguish between correct and incorrect images of wind spirals and wind barometers. I initially included a wide range of images from various angles, positions, lighting conditions, and backgrounds. However, through continued experimentation and research, I discovered that my dataset was not optimal for effectively training the model. The presence of extraneous objects around the objects of interest and incomplete images of the desired object reduced the accuracy of the model. Some images did not even contain the objects I was trying to detect.

After spending several weeks creating new datasets and running models on Edge Impulse, I turned to the TensorFlow example project that detects cats and dogs for inspiration. I noticed that the majority of images in the dataset contained either cats or dogs as the primary object of interest, even though the backgrounds were varied. This approach of including images with the object of interest occupying most of the frame without other distracting elements proved to be a more effective method for training the model(TensorFlow, n.d.).

As a result, I started a new project in TensorFlow that follows this approach. I included images of wind spirals and wind barometers with different backgrounds, but the focus of each image was on the object of interest. This approach has shown promising results, although the project is still ongoing, and further optimization may be required. (Images of the new dataset can be found below.)

# Images of the first attemps of the dataset (Edge impulse)
The following screenshot showcases a few bad examples of the different types of images that were uploaded to the Edge Impulse project
![Old Dataset](https://github.com/Gil-ADDA/CE-LAB/raw/75cd12f0f733c0bce10c60d37cd8ccc27e5b6eed/image/Old%20Dataset.png)


# Images of the new datasets (TensorFlow)
The accompanying screenshot displays several examples of the types of images uploaded to the TensorFlow project. Compared to the previous dataset, these images are cleaner and more focused on the object of interest.
![New Dataset](https://github.com/Gil-ADDA/CE-LAB/blob/48d980aa8a37915f5496a123c770a9b3924f68b3/image/New%20Dataset.png)


It's worth noting that Edge Impulse is a user-friendly and convenient tool. However, I ultimately switched to TensorFlow for this project because, at the time, some of the capabilities necessary for identifying two unique objects were not available to free users on Edge Impulse. I believed that the greater flexibility offered by TensorFlow would be more helpful in achieving my project goal.

# Screenshot and link to the TensorFlow project 

https://colab.research.google.com/drive/1fRwoDW84URBEveKRxg-vPBQK_DPzI-xO#scrollTo=ro4oYaEmxe4r 

![Tensorflow screenshot](https://github.com/Gil-ADDA/CE-LAB/blob/a4a19400f73572da09109843080afea15d62e0c2/image/Tensorflow%20project%20screenshot.png)

# Video Preview of the Model 
In the video demonstration, the image classification model is shown in action. As the camera is directed towards either the wind barometer or the wind spiral, the model quickly detects the object and labels it underneath the camera frame. However, if the camera is not directed towards either object, the model labels it as "unknown". This indicates that the model did not receive sufficient visual input to confidently classify the object with at least 80% accuracy. 

[![Video Thumbnail](https://img.youtube.com/vi/nQ7Ruwu12t8/0.jpg)](https://www.youtube.com/watch?v=nQ7Ruwu12t8)


# Workspace
Edge Impulse 
<br> <img src="https://www.edge-ai-vision.com/wp-content/uploads/2021/05/logo_edgeimpulse_may_2021.png" width="80" height="60">

In this project, Edge Impulse was utilized in three key ways. Firstly, the platform was used to store, tag, and split the dataset for efficient management. Secondly, it was utilized for building, training, and testing the machine learning model using various available algorithms. Finally, the trained model was deployed on an iPhone 12, enabling the execution of the model locally on the device


Google Sheets <br> <img src="https://www.gstatic.com/images/branding/product/1x/sheets_2020q4_32dp.png" width="60" height="60"> (Used for documenting the results of training and testing)

In this project, Google Sheets was used to document the experimentation process on Edge Impulse. This included documenting all the tweaks made to the models, as well as testing various different models during the training phase. The results of these tests were also documented, along with the results of the testing phase. By using Google Sheets to track and document the experimentation process, it allowed for a more organized and efficient workflow, making it easier to keep track of the various experiments performed on Edge Impulse.

# Challenges faced during development
Obtaining accurate results proved difficult during development. Additionally, finding the right dataset size was an issue; only 20 images were initially used and more were added as time went on, including images without the object and changes to the background. To gain a better understanding of machine learning, I employed various techniques and asked the Edge Impulse community for guidance (Attached underneath this paragraph). I studied other projects, articles, and books for additional insights. I also tweaked the model's variables, like architecture and image size, to boost performance.
Moreover before considering the question of how much data is required for your personal project, it is suggested to read the chapter on "Estimating Data Requirements" in the book "AI at the Edge" by Jason M. Pittman (2021) 
![Edge impusle community](https://github.com/Gil-ADDA/CE-LAB/blob/af24f8076a5d3730215290e521f82f2a187943f6/image/Edge%20impulse%20community.png)

The Link to the forum [https://forum.edgeimpulse.com/t/improve-performance-of-an-object-detection/6922/7](https://forum.edgeimpulse.com/t/improve-performance-of-an-object-detection/6922/7)

# Critical Reflection and Lessons Learned

Transfer learning can be used for both image classification and object detection, but image classification was more suitable for this project than object detection. Initially, the bounding box feature of object detection was appealing, but I lacked the resources to create a comprehensive dataset and the time to implement it. For those considering object detection, consider the time and resources needed to match the desired outcome with the data. For data requirements, refer to the table ,from chapter 7, and explanation in AI at the Edge (O'Reilly Media, Inc., 2023) (attached below).

![Data requirmentes Table](https://github.com/Gil-ADDA/CE-LAB/blob/1591c633fefef7717f39c174b17560ccd6428153/image/Data%20requirements%20for%20common%20tasks.png)


# Future work

For future work, there are several steps that could be taken to improve the project. One possibility is to create a  project in TensorFlow that is better able to detect both the wind spiral and wind barometer objects without relying on previous datasets. In addition, to the existing images, I'm thinking to add another type of image that includes examples of both objects in the same image. This can help to improve the accuracy of the transfer learning model and its ability to recognize and classify both objects together.

To use this new dataset to improve the object detection model, transfer learning can be employed. This involves fine-tuning a pre-trained model on a new dataset. By doing this, the model can learn to recognize the features of both objects together, improving its accuracy on the task of object detection.

### The link to the original Sheet (Object detection and transfer learning)  https://docs.google.com/spreadsheets/d/1LEAihPLvVjIKXl6ZXBGP7pdwTo1F_FSEUE8_gQM_ccs/edit?usp=sharing

## Contact Details
[<img src="https://img.icons8.com/color/48/000000/gmail.png"/>](mailto:giloo1047@gmail.com)
[<img src="https://img.icons8.com/color/48/000000/linkedin.png"/>](https://www.linkedin.com/in/gil-adda-16385510b/)


## Refrences
Edge Impulse. (n.d.). Object detection. Retrieved March 18, 2023, from https://docs.edgeimpulse.com/docs/tutorials/object-detection

Situnayake, D. and Plunkett, J. (2023). Estimating Data Requirements. In: AI at the Edge: A Developer's Guide. O'Reilly Media, Inc. Available at: https://learning.oreilly.com/library/view/ai-at-the/9781098120191/ch07.html#idm45988813512176 (Accessed: April 25, 2023)

Situnayake, D. and Plunkett, J. (2023). How to Build a Dataset: The Ideal Dataset, Balanced. In: AI at the Edge. O'Reilly Media. Available at: https://learning.oreilly.com/library/view/ai-at-the/9781098120191/ch07.html#idm45988813512176 (Accessed: 24 April, 2023).

TensorFlow. (n.d.). Transfer learning and fine-tuning. Retrieved from https://www.tensorflow.org/tutorials/images/transfer_learning 
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

