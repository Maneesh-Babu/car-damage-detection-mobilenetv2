# Car Damage Detection Model using Deep Learning
A deep learning project to classify car images as "damaged" or "whole" using MobileNetV2 and TensorFlow. Includes training, testing, and prediction scripts.
## Objective
Develop an AI Model using MobileNetV2, Tensorflow, and Keras to train it with car images classifying as 'damaged' and 'whole'. The goal is to make it accurately differentiate between a damaged and undamaged car.
## Dataset
The dataset is publicly available in Kaggle and must be extracted in the following structure:
car_data/data1a/ ├── training/ │ ├── 00-damage/ │ └── 01-whole/ 
                 └── validation/ ├── 00-damage/ └── 01-whole/
## Model
- Uses **MobileNetV2** as the feature extractor
- Custom head: GlobalAveragePooling2D + Dense + Dropout + Softmax
- Trained using **Adam optimizer** and **categorical crossentropy**
- Epochs: 5 (can be increased for better performance)
- Accuracy improves with more data and fine-tuning

## Files

| File | Purpose |
|------|---------|
| `train.py` | Trains and saves the model |
| `predict.py` | Loads model and predicts image class |
| `car_damage_model.h5` | Saved trained model |
| `dam1.jpeg`, `whole1.jpeg` | Example test images |
| `requirements.txt` | Required Python packages |

## Dataset Setup
This project uses a Car Damage Detection Dataset that contains images of damaged and undamaged (whole) vehicles, split into training and validation sets.

### Download Instructions
To keep the repository size small, the dataset is not included in this GitHub repo. You need to download it manually:

Visit the dataset source on Kaggle:
[Car Damage Detection Dataset](https://www.kaggle.com/datasets/anujms/car-damage-detection)

Download the ZIP file — usually named something like car-damage-detection.zip.
Extract it and organize it like this inside your project directory:
car_data/
└── data1a/
    ├── training/
    │   ├── 00-damage/
    │   └── 01-whole/
    └── validation/
        ├── 00-damage/
        └── 01-whole/


