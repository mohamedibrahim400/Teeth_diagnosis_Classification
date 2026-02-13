🦷 Teeth Classification using CNN (TensorFlow / Keras)

This project implements a Convolutional Neural Network (CNN) model to classify dental images into multiple categories using TensorFlow and Keras.
The model is trained with data augmentation and evaluated on separate validation and test datasets.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📌 Project Overview

The goal of this project is to build a deep learning model capable of classifying teeth images into different categories using a custom CNN architecture.

The workflow includes:

Data preprocessing & augmentation

Loading dataset using ImageDataGenerator

Visualizing class distribution

Displaying original & augmented images

Building a CNN model

Training & validation

Testing & evaluation

Saving & loading the trained model

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📂 Dataset Structure

The dataset is organized in the following directory structure:

Teeth_DataSet/
│
└── Teeth_Dataset/
    ├── Training/
    ├── Validation/
    └── Testing/


Each folder contains subfolders representing the class labels.

Unwanted folders (e.g., out, output, outputs) are excluded during loading.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

⚙️ Data Preprocessing

Images resized to: 128x128

Batch size: 32

Pixel normalization: rescale=1.0/255

Data augmentation techniques:

Rotation

Width & height shifting

Shearing

Zooming

Horizontal flipping

This improves generalization and reduces overfitting.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📊 Data Visualization

The project includes:

📈 Class distribution plots using seaborn

🖼 Display of original training images

🔄 Display of augmented images

This helps in understanding dataset balance and augmentation impact.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧠 Model Architecture

The CNN model consists of:

3 Convolutional layers:

Conv2D (32 filters)

Conv2D (64 filters)

Conv2D (128 filters)

MaxPooling after each convolution

Flatten layer

Dense layer (128 neurons)

Dropout (0.5) to reduce overfitting

Output layer with softmax activation

Loss Function:

categorical_crossentropy


Optimizer:

Adam


Metric:

Accuracy

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🚀 Training

Trained for 100 epochs

Validation dataset used during training

Accuracy and loss plotted for:

Training set

Validation set

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧪 Model Evaluation

The trained model is evaluated on a separate test dataset:

Test Accuracy: XX.XX

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

💾 Model Saving & Loading

The trained model is saved as:

teeth_classification_model.h5


It can be reloaded using:

load_model("teeth_classification_model.h5")

🛠 Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

Seaborn

OpenCV

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📌 Future Improvements

Use Transfer Learning (e.g., MobileNet, ResNet)

Add EarlyStopping & ModelCheckpoint

Hyperparameter tuning

Confusion matrix & classification report

Convert model to TensorFlow Lite for deployment
