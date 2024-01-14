# Image Classification Project README

## Overview
This project focuses on image classification using a pre-trained ResNet50 model with TensorFlow/Keras. The task is to classify images into one of four categories: Benign, [Malignant] Pre-B, [Malignant] Pro-B, [Malignant] early Pre-B.

## Model Architecture
We utilized the ResNet50 architecture as the base model and added a few layers on top for fine-tuning. The final layer is a Dense layer with a softmax activation function to output probabilities for each class.

Input
   |
[ Conv2D, BatchNorm, ReLU ]
   | 
[ Conv2D, BatchNorm, ReLU ]
   |
[ MaxPooling ]
   |
[ Residual Block ]
   |-------------------------[ Identity Block ]
   |                         |-----------------[ Identity Block ]
   |                         |
   |                         |-----------------[ Identity Block ]
   |
[ Global Average Pooling ]
   |
[ Fully Connected Layer ]
   |
Output (Class Predictions)


## training
We trained the model using a categorical cross-entropy loss function and the Adam optimizer with a custom learning rate schedule. The model was trained for 2 epochs on the provided data.

## Model Architecture
We utilized the ResNet50 architecture as the base model and added a few layers on top for fine-tuning. The final layer is a Dense layer with a softmax activation function to output probabilities for each class.

## Save Checkpoints to Google drive 


```python
# Model Architecture
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense
from tensorflow.keras.models import Model

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
predictions = Dense(4, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)





# Training Configuration
initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=40,
    decay_rate=0.96,
    staircase=False)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
              metrics=['accuracy'])

# Training
history = model.fit(
    x=train_gen,
    epochs=2,
    validation_data=valid_gen,
    steps_per_epoch=None,
    workers=2,
    callbacks=[cp_callback]
)



## Categories

The model is trained to classify images into the following categories:

Benign
[Malignant] Pre-B
[Malignant] Pro-B
[Malignant] early Pre-B


Feel free to modify the content based on your specific project requirements and information.