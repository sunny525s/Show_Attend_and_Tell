# Show, Attend and Tell: Neural Image Caption Generation with Visual Attention

### CS 4782 Final Project

### Sunny Sun, Michael Wei, Tony Chen, Linda Hu, Jiye Baek

## Overview

This project implements a neural image caption generation model that employs a visual attention mechanism to dynamically focus on different parts of an image while generating descriptive captions. The approach follows the paper, "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" by Xu et al.

## Installation and Setup

1. **Prerequisites:**

   - Ensure you have Python (version 3.10 or later) installed.
   - It is recommended to use a virtual environment to manage dependencies.

2. **Install Dependencies:**

   - Install the required Python libraries using:
     ```bash
     pip install -r requirements.txt
     ```

3. **Download Flickr8k Dataset:**
   - Download the image ZIP file from [Flickr8K](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip).
   - Extract its contents and rename the extracted folder to `dataset`.

## Running the Project

After setting up the dataset and installing the dependencies, you can start training the model:

1. **Prepare the Data:**

   - Run the following command to generate the training and validation splits:
     ```bash
     python src/data_prep.py
     ```

2. **Train the Model:**

   - Start training with:
     ```bash
     python src/main.py
     ```

3. **Inference (Caption Generation):**
   - To view the generated captions for a given image, run:
     ```bash
     python src/inference.py
     ```
