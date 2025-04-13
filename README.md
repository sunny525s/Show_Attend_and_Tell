# Show, Attend and Tell: Neural Image Caption Generation with Visual Attention

### CS 4782 Final Project

## Overview
This project implements a neural image caption generation model that employs a visual attention mechanism to dynamically focus on different parts of an image while generating descriptive captions. The approach follows the paper, "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" by Xu et al.

## Dataset Preparation

### 1. Download Flickr8k Annotations
- Download the dataset JSON file from [Stanford's webpage](https://cs.stanford.edu/people/karpathy/deepimagesent/).
- Place the downloaded JSON file in the directory:
  ```
  data/flickr8k
  ```

### 2. Preprocess the Data
- Run the following command to generate the training and validation splits:
  ```bash
  python data_preprocessing.py
  ```

### 3. Download Flickr8k Images
- Get the Flickr8k images from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k?resource=download).
- Extract the images and place them in the directory:
  ```
  data/flickr8k/imgs
  ```

## Installation and Setup

1. **Prerequisites:**
   - Ensure you have Python (version 3.7 or later) installed.
   - It is recommended to use a virtual environment to manage dependencies.

2. **Install Dependencies:**
   - Install the required Python libraries using:
     ```bash
     pip install -r requirements.txt
     ```

## Running the Project

After setting up the dataset and installing the dependencies, you can start training the model with:
```bash
python train.py
```
