
## CE-LAB  Image Classification Model Using Edge Impulse
<p align="center">
  <a href="https://github.com/Gil-ADDA/CE-Lab-project-detection-with-image-classification">
    <h1><a href="https://github.com/Gil-ADDA/CE-Lab-project-detection-with-image-classification">Github Repository Link</a></h1>
  </a>
</p>
<a href="https://github.com/Gil-ADDA/CE-Lab-project-detection-with-image-classification">https://github.com/Gil-ADDA/CE-Lab-project-detection-with-image-classification</a>
# Project Overview

This project was inspired by the need for better engagement with the devices and projects in the Connected Environment Lab, both for students and guests visiting the lab. The primary objective is to detect two unique objects in the lab - the wind spiral and wind barometer - using an image classification program. By successfully identifying these objects, users can better interact with and learn about the lab's devices.

The project employs a transfer learning approach to train an object detection model for detecting the wind spiral and wind barometer objects. A dataset consisting of 149 images, representing the two object labels, was used for training the model. Transfer learning leverages a pre-trained model to fine-tune the object detection task, helping to achieve better performance despite the smaller dataset.

The object classification model can be accessed by students and guests in the lab to help them identify the wind spiral and wind barometer objects. This interactive experience aims to foster better engagement and understanding of the devices in the lab, benefiting both students and guests. Possible future work includes improving the model's performance by collecting a more diverse dataset and incorporating an object detection model.

# problem statement 
The Connected Environment Lab has a variety of unique devices and projects that can be difficult for students and visitors to engage with and understand. In particular, the wind spiral and wind barometer objects are visually impressive but require explanation to fully appreciate. The goal of this project is to develop an image classification program using transfer learning that can accurately detect and identify these objects in order to enhance engagement and understanding of the devices in the lab. The challenge is to achieve high accuracy with a relatively small dataset.

![WindS Objects](https://github.com/Gil-ADDA/CE-LAB/blob/7ab792b29093a8fc4cc896c16054d685e39c08ff/image/WindS%20Objects.png)
# Research question 

The research question for this project was initially formulated as "What is the impact of using an image classification program on increasing engagement with the devices in the Connected Environment Lab?" However, based on feedback from the teaching staff, the question was revised to be more focused on the requirements of the course. The new research question is: 
How can transfer learning be utilized to enhance the accuracy of classifying wind spiral and wind barometer using computer vision? (The two unique objects in the Connected Environment Lab). 

# Declaration

This project started as an object detection project, aiming to detect two unique objects in the Connected Environment Lab - the wind spiral and wind barometer. However, due to the limited dataset, it was challenging to achieve satisfactory results using traditional object detection techniques.
Throughout the project, the terms "object detection" and "image classification" may be used interchangeably, but the fundamental goal remains the same - to accurately detect and classify the wind spiral and wind barometer objects. The choice to utilize image classification over object detection was made to achieve better performance with the small dataset. 
*see the diffrences in the image below(adapted from DataCamp, 2018).

![Exmple from DataCamp](https://github.com/Gil-ADDA/CE-LAB/blob/ef14ef0d8d21c367b88908499b6ad4e75ce8f532/image/Object-D-VS-Image-Class.jpeg)




# Background 
The project was motivated by the need to improve engagement with the devices and projects in the Connected Environment Lab, both for students and guests. The primary objective is to detect two unique objects in the lab - the wind spiral and wind barometer - using an image classification program. The lab receives a lot of visitors, and with the increasing number of devices and projects, it can be challenging for visitors to understand the narrative and functionality of each project. The wind spiral and wind barometer are two examples of devices that visitors find intriguing but may not understand fully without the help of a teaching staff member. The project's short-term benefits include allowing guests to interact with and learn about these devices in a more meaningful way. The long-term goal is to create a virtual hub of all the projects in the lab, which will be open to everyone. Students and guests can connect with the co-creators of the projects and consult with them, potentially leading to the evolution of these projects or the creation of similar ones. The dataset used for this project consists of 149 images for both wind spiral and wind barometer classes. While working on the project, I faced challenges in achieving good results and figuring out the right amount of data required for the project to work. After experimenting with different approaches, I decided to pivot to transfer learning based on recommendations from the book Algorithms for Edge AI. 

![Object detection bounding box attempts](https://github.com/Gil-ADDA/CE-LAB/blob/d6dc0140d765485b9b29957252cfe8582ada2612/image/Object%20detection%20bounding%20box%20attempts.png)

# Getting Started Image classification Model Using Transfer Learning
## 1. Dataset Collection and Split
To begin, I collected a dataset of 149 images, including two labels representing the unique objects in the lab - the wind spiral and wind barometer. I then divided the dataset into training and test sets, following a 77% training and 23% test split, which aligns with the commonly recommended 80% training and 20% test split. In the training set, there were 74 images of wind spirals and 40 images of wind barometers, while the test set had 11 images of wind barometers and 24 images of wind spirals.

Based on feedback from teaching staff, it was recommended that an equal number of images be included for each object in both training and test datasets. This would help ensure the model is trained and tested on each object equally and improve overall accuracy.  To provide optimal training and testing of the model, having an equal number of images for each object in the training and test datasets. Moreover, I found that this advice is supported by the "The Ideal Dataset" section of the book "AI at the Edge," which emphasizes the importance of having balanced datasets to avoid bias in the model(O'Reilly Media, Inc., 2023).

![split of the data](https://github.com/Gil-ADDA/CE-LAB/blob/5dd489074ee6368cd160508a72b5791f405d4a10/image/split%20of%20the%20data.png)

## 2. Image Upload and Preprocessing block Configuration 
After collecting the images, I uploaded them to the Edge Impulse platform for further processing. To prepare the images for model training, I first resized them to a standardized 96x96 pixel size, and applied preprocessing and normalization techniques to the image data. To maintain consistency, I kept the images' color depth as RGB. 
To further prepare the image data for model training, I created a processing block within the Edge Impulse platform. This block preprocessed and normalized the images, making it easier for the model to process the images and facilitating faster detection of the wind spiral and wind barometer objects.

![Edge-Impulse learning block](https://github.com/Gil-ADDA/CE-LAB/blob/0ef6f2352801a14fb946319877d21ce1ab913700/image/Edge%20impuse%20learning%20block%20interface.png)


## 3. Data Exploration and Feature Visualization
Throughout the process, Edge Impulse's Data Explorer tool was used to identify the characteristics of each image and how the model categorized them. Data Explorer shows all project data and was created by passing input through a pre-trained visual model. This allows for fast analysis of the data, including feature distributions and relations between each of the images. Pre-trained model data exploration has advantages like faster results and improved performance compared to training from scratch. When data sets are small, this pre-trained model knowledge can be adjusted to suit the data.

![Data Explorerr](https://github.com/Gil-ADDA/CE-LAB/blob/70664a08f860f2fc9c783fa6262c89780430bca4/image/Data%20Explorerr.png)

## 4. Model Selection
To improve the performance of the image classification task, I employed transfer learning with the MobileNetV2 96x96 1.0 model, which was pre-trained on a large dataset and fine-tuned on my small dataset. This approach proved effective in achieving better results even with a limited amount of data. Additionally, the comparison of the experimentation results of the transfer learning process can be found in the [Model Training](#5-model-training) section of this project's repository.

Moreover, based on the Model Architectures section in Chapter 4 of the book "AI at the Edge" by O'Reilly Media, I would recommend sticking with the MobileNetV2 architecture as it is suitable for deployment on mobile devices. The pre-trained MobileNetV2 model provides an excellent starting point for transfer learning and can be fine-tuned to achieve better performance on specific tasks.

## 5. Model Training
I trained the model using the processed images and learning block configuration with a batch size of 32 and 40 epochs. To address overfitting issues, I modified the Keras expert mode settings, setting the learning rate to 0.0015. I didn't use the auto-balance dataset or data augmentation options in this case, as I was not aware of the need to add them manually in the Keras code (Expert mode) at the time. Despite this, the model testing results showed an accuracy of 94.29%, indicating that the model is effective in identifying the wind spiral and wind barometer objects. However, it's important to consider using data augmentation and class balancing techniques in future projects, especially when working with a small dataset like in this project. Data augmentation can help prevent overfitting by artificially increasing the size of the training dataset and adding diversity to the data. Class balancing can ensure that the model has equal representation of all classes, even when one class has significantly more samples than another.

After experimenting with various hyperparameters, I found that a dropout rate of 0.3 was most suitable for my model and helped to prevent overfitting. It can be seeing in the code the BatchNormalization layer, which was used to improve performance of the model and make it more stable. This layer helps to normalize the activations of the previous layer and reduces the internal covariate shift, which can lead to faster training and better accuracy.

The key parameter of the model: 
* MobileNetV1.0_1.96x96.color
After testing multiple model architectures and configurations, I chose MobileNetV1.0_1.96x96.color for image classification as it showed the best accuracy and performance.

* BATCH_SIZE: determines the number of samples used per training iteration, affecting the speed and memory of the process. A value of 32 optimizes these factors. 

* EPOCHS: number of passes through the dataset, impacting convergence. 40 is chosen to capture complex patterns. 

* LEARNING_RATE: the size of each training step, affecting speed and accuracy. A value of 0.0015 strikes a balance.

* FINE_TUNE_EPOCHS: number of additional passes to fine-tune the model and increase accuracy. 10 are used in this model. (After the 40 epochs for improvment) 

* FINE_TUNE_PERCENTAGE: percent of base layers to fine-tune. 65% is used to balance adaptability with pre-trained knowledge.

* max_delta: variability in brightness for data augmentation. A value of 0.2 increases model abilities to deal with different lighting conditions.

![Transfer Learning Documentation](https://github.com/Gil-ADDA/CE-LAB/blob/78db88dd2031858951dd2e66a405a8f9987847d3/image/transfer%20learning%20documention%20.png)

![Training Data](https://github.com/Gil-ADDA/CE-LAB/blob/4bc06f64ee0e758ea9d4a4ba2524ebef6b4640e2/image/Training%20Data.png)


## 6. Model Testing and Result

After training the image classification model with transfer learning, I tested it with 23% of the dataset and achieved an accuracy of 93.5%. To determine the optimal validation set size, I experimented with various parameters and found good performance with a size of 40. I classified all the test images using the model and set the threshold at 0.8 (80%). In March 2022, EON Tuner was only available for enterprise use, which may be useful for identifying appropriate architecture for ML projects.

It's worth noting that while this model performed well, there is always a risk of overfitting when training on a small dataset. To address this, I plan to further validate the model on additional datasets in the future.
![Result of the Model](https://github.com/Gil-ADDA/CE-LAB/blob/fe0ba971f1ac7ddcd916a45734f81571d11a0c33/image/Result%20of%20the%20model1.png)

### Confusion matrix 
In the *confusion matrix above* it can be seen that the model performed well, achieving an overall F1 score of 0.95. It accurately identified all wind barometer images in the validation set with 100% precision. However, it misclassified 7.7% of the wind spiral images as wind barometer, resulting in 92.3% recall. Moreover the performance of the model may vary with different datasets and evaluation metrics. It's important to note that the 100% precision achieved in classifying all wind barometer images in the validation set indicates that it has been trained to accurately identify features of these images. However, this is likely due to the small size of the dataset. It's possible that with a larger and more diverse dataset, the precision score could decrease.

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


In this project, Edge Impulse was utilized in three key ways. Firstly, the platform was used to store, tag, and split the dataset for efficient management. Secondly, it was utilized for building, training, and testing the machine learning model using various available algorithms. Finally, the trained model was deployed on an iPhone 12, enabling the execution of the model locally on the device.


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
In order to improve this project, there are several steps that can be taken.
First, a larger dataset can be created for each object, with at least 300 images for each object and ensuring that the object is the main part of the image (occupying more than 70% of the image). To improve training efficiency and accuracy, transfer learning with image classification can be used. Additionally, a new chunk of dataset can be created with both objects in one image, which can help to improve the model's ability to recognize and classify both objects together. Given the success of the MobileNet model architecture for this project, it will be used going forward. 

In order to make the project accessible to new students and visitors, a web application will be developed that allows users to easily detect and identify the objects in the CE-Lab using their smartphones.

The web page files for this project are included in this repository as well. These files contain the HTML, CSS, and JavaScript (templates) code that is used to create the web application for the “CE-lab object detection”.
Additionally, the vision of the user interface of the app is attached in the image underneath designed in Hotpot<a href="https://hotpot.ai/" target="_blank"><img src="https://img.shields.io/badge/-hotpot-181717?style=flat-square&logo=GitHub&logoColor=white"></a> that provides an overview of how the model will look in action.



![WEB-APP](https://github.com/Gil-ADDA/CE-LAB/blob/468516b6af0e46fc6519c8d275403c36a74bc69b/image/WEB%20APP%20CE-LAB.png)


### The link to the original Sheet (Object detection and transfer learning)  https://docs.google.com/spreadsheets/d/1LEAihPLvVjIKXl6ZXBGP7pdwTo1F_FSEUE8_gQM_ccs/edit?usp=sharing

## Contact Details
[<img src="https://img.icons8.com/color/48/000000/gmail.png"/>](mailto:giloo1047@gmail.com)
[<img src="https://img.icons8.com/color/48/000000/linkedin.png"/>](https://www.linkedin.com/in/gil-adda-16385510b/)


## Refrences

Edge Impulse. (n.d.). Object detection. Retrieved March 18, 2023, from https://docs.edgeimpulse.com/docs/tutorials/object-detection

Situnayake, D. and Plunkett, J. (2023). Estimating Data Requirements. In: AI at the Edge: A Developer's Guide. O'Reilly Media, Inc. Available at: https://learning.oreilly.com/library/view/ai-at-the/9781098120191/ch07.html#idm45988813512176 (Accessed: April 25, 2023)

Situnayake, D. and Plunkett, J. (2023). How to Build a Dataset: The Ideal Dataset, Balanced. In: AI at the Edge. O'Reilly Media. Available at: https://learning.oreilly.com/library/view/ai-at-the/9781098120191/ch07.html#idm45988813512176 (Accessed: 24 April, 2023).

TensorFlow. (n.d.). Transfer learning and fine-tuning. Retrieved from https://www.tensorflow.org/tutorials/images/transfer_learning 
## Appendix

DataCamp. (2018, April). Object Detection Guide. Retrieved April 25, 2023, from https://www.datacamp.com/tutorial/object-detection-guide

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

