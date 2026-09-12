\# EN3150 Assignment 03: Resource-Constrained CNN for Edge Image Classification


\## Group Members

\*   \*\*Manitha Ayanaja\*\* - \[Insert Index Number] - University of Moratuwa

\*   \*\*\[Team Member 2 Name]\*\* - \[Insert Index Number]

\*   \*\*\[Team Member 3 Name]\*\* - \[Insert Index Number]

\*   \*\*\[Team Member 4 Name]\*\* - \[Insert Index Number]



\## Project Overview

This repository contains the implementation of a lightweight image classifier designed for resource-constrained edge devices\[cite: 1]. The project explores the trade-offs between accuracy, memory footprint, and computational cost by comparing custom CNN architectures against State-of-the-Art (SOTA) lightweight networks\[cite: 1].



\### Implemented Models

\*   \*\*Model A (Standard CNN):\*\* A baseline model using standard 2D convolutions and max-pooling layers\[cite: 1].

\*   \*\*Model B (Lightweight CNN):\*\* A highly optimized network utilizing Depthwise Separable Convolutions, strictly constrained to under 100,000 trainable parameters\[cite: 1].

\*   \*\*SOTA Models:\*\* Fine-tuned pre-trained mobile networks (e.g., MobileNetV2, SqueezeNet, or EfficientNet-B0) for performance comparison\[cite: 1].



\## Setup and Execution

1\.  Clone this repository to your local machine.

2\.  Install the necessary Python dependencies by running: `pip install -r requirements.txt`

3\.  Place your chosen UCI Machine Learning Repository dataset inside the `data/` directory\[cite: 1].

4\.  Run `src/data\_prep.py` to automatically downscale the images to $64 \\times 64$ pixels and generate the 70/15/15 train/val/test splits\[cite: 1].

5\.  Execute `src/train\_eval.py` to train the models for a minimum of 20 epochs and output the accuracy, precision, recall, and confusion matrices\[cite: 1].

