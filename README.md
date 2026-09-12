# EN3150 Assignment 03: Resource-Constrained CNN for Edge Image Classification

## Group Members
*   **Manitha Ayanaja** - [Insert Index Number] 
*   **[Team Member 2 Name]** - [Insert Index Number]
*   **[Team Member 3 Name]** - [Insert Index Number] 
*   **[Team Member 4 Name]** - [Insert Index Number] 

## Project Overview
This repository contains the implementation of a lightweight image classifier designed for resource-constrained edge devices (e.g., microcontrollers, Raspberry Pi). The project explores the trade-offs between accuracy, memory footprint, and computational cost by comparing custom CNN architectures against State-of-the-Art (SOTA) lightweight networks.

## Implemented Models
*   **Model A (Standard CNN):** A baseline model using standard 2D convolutions and max-pooling layers.
*   **Model B (Lightweight CNN):** A highly optimized network utilizing Depthwise Separable Convolutions, strictly constrained to under 100,000 trainable parameters.
*   **SOTA Models:** Fine-tuned pre-trained mobile networks (e.g., MobileNetV2, SqueezeNet) for performance comparison.

## Dataset
*   **Source:** [Insert Dataset Name from UCI Machine Learning Repository]
*   **Preprocessing:** Images are downscaled to a maximum resolution of 64x64 pixels to emulate a low-resource sensor environment.
*   **Splits:** 70% Training / 15% Validation / 15% Testing.

## Setup and Execution

1.  **Clone this repository:**
    ```bash
    git clone [Insert your GitHub Repository URL here]
    cd EN3150_Assignment_03
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare the Data:** 
    Place your downloaded dataset in the `data/` directory and run the preprocessing script to downscale and split the images.
    ```bash
    python src/data_prep.py
    ```

4.  **Train and Evaluate:** 
    Run the training pipeline to generate models, plot loss curves, and output evaluation metrics (accuracy, precision, recall, confusion matrix).
    ```bash
    python src/train_eval.py
    ```

## Results & Trade-off Analysis
*(To be filled after training and evaluation)*

| Model | Parameter Count | Model Size | Training Time/Epoch | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Model A (Standard)** | - | - | - | - |
| **Model B (Lightweight)** | < 100k | - | - | - |
| **SOTA Model 1** | - | - | - | - |
| **SOTA Model 2** | - | - | - | - |
