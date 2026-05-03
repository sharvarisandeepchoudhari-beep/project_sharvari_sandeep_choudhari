# ASL Number Image Classifier

This project implements a deep learning model to classify American Sign Language (ASL) hand gestures representing numbers (0–9). The model is trained using a convolutional neural network based on ResNet18.

---

## Project Structure

* `model.py` → Model architecture
* `train.py` → Training loop
* `dataset.py` → Dataset loading and preprocessing
* `predict.py` → Prediction logic
* `interface.py` → Interface for running predictions
* `config.py` → Hyperparameters and paths
* `data/` → Sample dataset
* `checkpoints/final_weights.pth` → Trained model weights

---

## Installation

Install the required dependencies:

pip install torch torchvision numpy pillow

---

## Training

To train the model from scratch:

python train.py

This will train the model and save weights to:

checkpoints/final_weights.pth

---

## Prediction

To run the prediction pipeline:

python interface.py

---

## Manual Test (Optional)

You can also test the model directly using:

python -c "from interface import *; import torch; from config import NUM_CLASSES; model=TheModel(NUM_CLASSES); model.load_state_dict(torch.load('checkpoints/final_weights.pth', map_location='cpu')); model.eval(); print(the_predictor(model, ['data/0/hand1_0_bot_seg_1_cropped.jpeg']))"

---

## Dataset

The dataset used for this project was sourced from Kaggle.
Only a subset of the dataset is included in this repository for demonstration purposes.

---

## Notes

* The model is trained on a small dataset for demonstration.
* Accuracy is not the primary focus; correct implementation and functionality are emphasized.
* The project demonstrates a complete pipeline: data loading → training → saving weights → inference.

---

## Summary

This repository contains a fully functional ASL number classifier with:

* Working training pipeline
* Saved and loadable model weights
* End-to-end prediction capability

The project is ready to run and evaluate.
