# Skin-Cancer-Detection-using-TensorFlow
The goal is to automate skin cancer screening from dermoscopic images, which can assist early diagnosis and improve clinical outcomes.

🧠 Overview

Skin cancer detection from visual data is a critical task in medical imaging and diagnostics. In this project, we apply state-of-the-art deep learning methods to classify dermoscopic images of skin lesions into two categories: benign and malignant. 


📦 Dataset

The dataset consists of images categorized into two folders—benign and malignant cases—commonly sourced from public datasets like ISIC on Kaggle. The images are pre-organized in subdirectories to allow easy loading with Python scripts. 


🛠 Built With

Python

TensorFlow & Keras

NumPy

Pandas

Matplotlib

Scikit-learn

🚀 Key Steps
1. Install Dependencies
pip install tensorflow numpy pandas matplotlib scikit-learn

2. Load and Prepare Data

Recursively collect all image file paths.

Create a DataFrame mapping each image path to its label (benign or malignant).

Convert labels to binary (0/1).

Split into training/validation sets. 


3. Define Preprocessing Pipeline
def decode_image(filepath, label=None):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


Resize and normalize images for the model input. 


4. Build the Model using Transfer Learning
from tensorflow.keras.applications.efficientnet import EfficientNetB7
pre_trained_model = EfficientNetB7(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=False
)
for layer in pre_trained_model.layers:
    layer.trainable = False


Use EfficientNetB7 (ImageNet pretrained) as feature extractor. 


5. Train the Model
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['AUC']
)
history = model.fit(train_ds, validation_data=val_ds, epochs=5)


Train over multiple epochs observing validation performance. 


6. Visualize Training Progress

Plot training/validation loss and AUC curves to evaluate performance over time. 


📊 Results

You’ll see plots for:

Training and validation loss

Training and validation AUC

These metrics help assess overfitting and predictive quality. 


💡 Next Enhancements

Here are ideas for future improvements:

Data augmentation (rotation, flips, color jitter)

Additional performance metrics (precision, recall, F1)

Expand to multi-class classification using larger datasets

Deploy model using Flask, FastAPI, or Streamlit
