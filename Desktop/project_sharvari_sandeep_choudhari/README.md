# Image Classification using ResNet18

## Project Description
This project implements an image classification system using a deep learning model based on ResNet18. The model is fine-tuned using transfer learning to classify images into multiple categories from a custom dataset. The pipeline includes data loading, preprocessing, training, and inference.

---

## Model Choice
We use ResNet18 (pretrained on ImageNet) from torchvision.

Reason for choice:
- Strong performance on image classification tasks
- Works well with limited data due to transfer learning
- Residual connections help with stable training

The final fully connected layer is modified to output:
NUM_CLASSES = 10

---

## Dataset

The dataset must be structured in the following format:

data/
 ├── class_1/
 ├── class_2/
 ├── class_3/
 ├── ...
 └── class_10/

Each folder contains images of that class.

### Dataset Download
(Add your dataset link here)

Example:
https://your-dataset-link.com

### Setup Instructions
1. Download the dataset
2. Extract it into the project root directory
3. Ensure the folder name is:
   data/

---

## Installation

Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
