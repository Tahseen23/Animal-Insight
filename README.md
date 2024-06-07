# Animal Retrieval System

Welcome to the Animal Retrieval System project! This system allows users to upload images of animals, which are then processed to detect, crop, and classify the animals. The project utilizes YOLO (You Only Look Once) for object detection and VGG16 for image classification. The web interface, built using Flask, displays information about the animal, including its favorable locations, survival issues, and common habitats.

# Installation
To get a local copy of the project up and running, follow these steps:

# Prerequisites
1. Python 3.6 or higher
2. pip (Python package installer)
3. Virtual environment (recommended)

# Uploading an Image
1. Click on the "Upload Image" button.

![new](Screenshot(21).png)

2. Select an image of an animal from your local machine.
3. Click "Predict" to upload the image.

![final](<Screenshot(24).png>)

4. The system will process the image, detecting and cropping the animal using YOLO. The cropped image will then be classified using VGG16. The web interface will display information about the identified animal, including:
   - Name
   - Favorable location
   - Survival issues
    - Locations mainly found

# Technologies Used
- Flask: Web framework for Python.
- YOLO: Object detection algorithm used to detect and crop animals in images.
- VGG16: Convolutional Neural Network used for classifying the cropped animal images.
- OpenCV: Library for image processing.
- TensorFlow/Keras: Frameworks used to implement the deep learning models


<video controls src="Animal Insight - Google Chrome 2024-06-07 18-35-29.mp4" title="hello"></video>