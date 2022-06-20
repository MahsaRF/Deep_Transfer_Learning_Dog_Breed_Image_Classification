# Transfer_Learning-Based_Stanford_Dog_Breed_Picture_Classification_using_Keras

Transfer-Learning Based Stanford Dog Breed Picture Classification using Keras

In this project we utilized an Inception-Resnet based deep learning structure to classify the breed from the dog picture in a big dog image dataset. Dog breed categorization is a specific application of the convolutional neural networks because it is considered as a fine-grained image classification problem and on the other hand, we have a very small training example dataset and short computing power. From a deep learning perspective, the image classification problem can be solved through transfer learning. When we are repurposing a pre-trained model for our own needs, we start by removing the original classifier, and then we add a new classifier that fits our purposes. The main idea is to keep the convolutional base in its original form and then use its outputs to feed the classifier. We are using the pre-trained model as a fixed feature extraction mechanism, which can be useful if the computational power is short, our dataset is small, and/or pre-trained model solves a problem very similar to the one we want to solve. In this project, we have repurposed two of the best pre-trained image classification models named Inception-ResNet-V2 and Inception-V3 as a fixed feature extraction mechanism in the KERAS tool. We use the transfer learning with image augmentation to get better model generalization and achieved a validation accuracy of around 90% using the Inception-ResNet-V2 and a validation accuracy of around 80% utilizing the Inception-V3 model.

This project was submitted as part of the graduate level Machine Learning course taught by Professor Ricardo Vilalta at the University of Houston Fall 2019
