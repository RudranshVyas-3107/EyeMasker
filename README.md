# EyeMasker: An Efficient Mask Detection System


## Introduction

EyeMasker is a machine learning project developed to detect whether individuals in images are wearing face masks. It uses Python, OpenCV, and popular machine learning frameworks to build and train a Convolutional Neural Network (CNN) model. This project is structured to run efficiently in Google Colab, leveraging cloud-based resources for easy access and scalability.

## Key Features

- **Flexible Data Loading**: Easily import and preprocess image datasets from Google Drive.
- **Comprehensive Data Labeling**: Categorize images into distinct "with mask" and "without mask" classes.
- **Robust Model Training**: Build and train a deep learning CNN model specifically for mask detection.
- **Performance Evaluation**: Evaluate the model's performance with high precision using test data.
- **Real-Time Inference**: Make fast and accurate predictions on new, unseen images.

## Requirements

- Python 3.x
- Google Colab (for execution)
- Google Drive (for storing and loading datasets)
- Essential Libraries: OpenCV, TensorFlow/Keras, NumPy, os

## Setup Instructions

1. Clone this repository or download the project files to your system:
    ```bash
    git clone https://github.com/RudranshVyas-3107/EyeMasker.git
    ```
2. Open `mask_detection.ipynb` in Google Colab.
3. Install the required Python libraries using the following commands:
    ```python
    !pip install opencv-python
    !pip install tensorflow
    ```

## Usage Guide

1. **Mount Google Drive**: Use Google Colab to mount your Google Drive, making it easy to access your dataset.
2. **Data Preparation**: Load and label your dataset, ensuring images are correctly categorized into "with mask" and "without mask" groups.
3. **Train the Model**: Use the provided CNN architecture to train your model on the labeled dataset.
4. **Evaluate Model**: Assess the model's performance using validation and test datasets.
5. **Run Inference**: Use the trained model to predict mask usage on new, unseen images.

## Dataset Structure

Ensure that your dataset in Google Drive is organized as follows:
- `with_mask`: Contains images of individuals wearing face masks.
- `without_mask`: Contains images of individuals without face masks.

## Future Improvements

- **Model Optimization**: Explore different CNN architectures to enhance model accuracy.
- **Real-Time Detection**: Implement the system in real-time video streams using OpenCV.
- **Deployment**: Deploy the model on a cloud platform to allow for scalable, real-time mask detection services.

## Contributions

Contributions to improve EyeMasker are welcome! Feel free to fork this repository, make changes, and submit a pull request with your improvements or suggestions.

## Acknowledgements

This project leverages open-source libraries like OpenCV and TensorFlow. Special thanks to the creators of the datasets that made the training and evaluation of this model possible.
