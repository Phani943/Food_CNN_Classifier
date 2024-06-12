# Food CNN Classifier

This repository contains a convolutional neural network (CNN) model trained to classify images of different types of food, including pizza, burger, ice cream, samosa, and sushi. The model was developed with own architecture without using transfer learning.

## Project Overview

The goal of this project is to gain proficiency in deep learning and computer vision techniques by building a model capable of accurately classifying food images into predefined categories. The model can be used for various applications, such as food recognition in restaurant menus, dietary analysis, or food recommendation systems.

## Dataset

The dataset used for training was prepared by me and consists of approximately 2,000 images across five food categories: pizza, burger, ice cream, samosa, and sushi. The images were collected from various sources, including search engines, personal collections, and image scraping techniques. Each image was manually labeled with its corresponding food category to facilitate supervised learning.

The dataset underwent preprocessing steps, including resizing to a standard size of 256x256 pixels and normalization to ensure consistency and optimal training performance. Data augmentation techniques, such as random rotations, flips, and shifts, were also applied to augment the dataset and improve model robustness.

## Model Architecture

The CNN architecture consists of multiple convolutional and pooling layers followed by fully connected layers for classification. The model utilizes rectified linear unit (ReLU) activation functions and softmax activation in the output layer. Dropout regularization is applied to prevent overfitting.

## Training

The model was trained using the Adam optimizer with a categorical cross-entropy loss function. Training was performed for 50 epochs with early stopping based on validation accuracy and validation loss to prevent overfitting. The training process achieved a final training accuracy of approximately 78.2% and a validation accuracy of approximately 74.06%.

## Usage

To use the trained model for inference, simply load the model weights and pass input images through the network. Some example are given in the notebook demonstrating how to load and use the model for classification is provided in the repository.

## Repository Structure

- `data/`: Contains the dataset used for training.
- `notebooks/`: Contains example notebooks for training and inference.

## Future Work

- Fine-tune the model architecture and hyperparameters for improved performance.
- Explore additional data augmentation techniques to enhance model robustness.
- Investigate transfer learning approaches using pretrained models for feature extraction.
