# Image Classification Project README

## Overview
This project focuses on image classification using a pre-trained ResNet50 model with TensorFlow/Keras. The task is to classify images into one of four categories: Benign, Malignant Pre-B, Malignant Pro-B, Malignant early Pre-B.

We utilized the ResNet50 architecture as the base model and added a few layers on top for fine-tuning. The final layer is a Dense layer with a softmax activation function to output probabilities for each class.


## Resnet Blocks structure:

The images for the architecture are obtained from this [blog](https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33).

1. Identity Block:

<img src="images/identity_block.png" style="width:800px;height:300px;">

2. Convolutional Block:

<img src="images/convolutional_block.png" style="width:800px;height:300px;">

## Prerequisites
The dependent packages for MAP-NN are as follows:

* Python 3.9.12
* TensorFlow 2.15.0
* torch 2.1.2
* OpenCV 4.6.0
* Sklearn 1.26.3
* SciPy 1.7.3
* skimage 0.19.2
* keras 2.15.0
* pandas 1.4.2

## Usage
### Prepare the data
run
```
python data_prepare.py
```
python data_prepare.py lists the number of images, divides them into test and train sets, crops them to uniform dimension (224,224,3) segments and finally stores them to data_reply
### demostrate the data
run

```
python data_demo.py
```
### train the data
run
```
model_train.py
```


## Google colab version
There is a google colab version
run
```
classifiction_ResNet50.py
```
 To avoid the training interruption(which happens frequently in google colab train), all the processed data including checkpoints is saved in Google Drive

### import google drive
First call google drive 

```
python from google.colab import drive
drive.mount('/content/drive')
```
### Create a callback that saves the model's weights
```
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
history =         model.fit(x=train_gen,
                            epochs=2,
                            validation_data=valid_gen,
                            steps_per_epoch=None,
                            workers=2,
                            callbacks=[cp_callback]
                            )```
```



Feel free to modify the content based on your specific project requirements and information.