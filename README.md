# assignment_DL
Multiclass Fish Image Classification
This project focuses on training and deploying a deep learning model to classify fish images into multiple categories using popular convolutional neural networks (CNNs). It includes:
Colab for model training and evaluation
Streamlit web app for interactive prediction

# Project Structure
Multiclass_Fish_Image_Classification/
│
├── Multiclass_Fish_Image_Classification.ipynb   # Model training & evaluation
├── app.py                                       # Streamlit app for image classification
├── vgg16_best.h5                                # Trained VGG16 model
├── best_resnet_model.h5                         # Trained ResNet model
├── best_mobilenet_model.h5                      # Trained MobileNet model
├── best_inception_model.h5                      # Trained InceptionV3 model
├── best_efficientnet_model.h5                   # Trained EfficientNet model

# Project Overview
1. Model Training (Multiclass_Fish_Image_Classification.ipynb)
The colab notebook includes:
Data preprocessing and augmentation
Model building using transfer learning (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0)
Fine-tuning and saving the best models
Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
Model Saving: Best performing models are saved as .h5 files

2. Model Deployment (app.py)
The app.py is a Streamlit-based web application that allows users to:
Upload a fish image (.jpg, .jpeg, .png)
Choose one of the pre-trained models
Get real-time predictions with confidence scores

Dynamic Input Shape Handling:
The app automatically detects the input shape required by the selected model and resizes the uploaded image accordingly, ensuring compatibility with all models.

## How to Run the Project
Prerequisites
Install required Python libraries:
pip install streamlit tensorflow pillow numpy

# To Train Models (Optional)
Open and run Multiclass_Fish_Image_Classification.ipynb in colab Notebook to:
Preprocess the dataset
Train and evaluate models
Save the best model files

# To Run the Streamlit App
open the terminal 
type python -m venv env
activate the environment
pip install streamlit
streamlit run app.py
This will open a browser window with the fish classification interface.

# Pre-trained Models Used
The following pre-trained models were fine-tuned for the classification task:
VGG16
ResNet50
MobileNet
InceptionV3
EfficientNetB0

Fish Classes
The classification covers 11 fish classes, referred to as:
Class_1, Class_2, ..., Class_11
You can update these names with actual fish species if available.

# Example Usage
Launch the app.
Upload an image of a fish.
Select a model from the sidebar.
View predicted class and confidence score.

# Key Features
Multiple model support
Input shape auto-detection
Real-time predictions
Easy deployment with Streamlit
