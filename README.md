# 🎭 Expression-DCGAN: Landmark-Conditioned Facial Generation

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/sepehrbayatnezhad/expression-dcgan)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C)
![dlib](https://img.shields.io/badge/dlib-Facial_Landmarks-green)

## 📖 Project Overview
This project implements a **Conditional Generative Adversarial Network (cGAN)** designed to synthesize realistic human facial expressions from structural facial landmark maps. 

Using the **JAFFE (Japanese Female Facial Expression)** dataset, we first extract a 68-point "dot skeleton" (landmark map) from real faces. The GAN is then trained on these paired spatial images to learn the mapping between the geometric structure of an emotion (where the lips, eyebrows, and eyes are positioned) and the photorealistic rendering of a human face.

## 🧠 Core Concept: Image-to-Image Translation
Standard GANs generate random images from a vector of random noise. **Conditional GANs (cGANs)**, specifically architectures like **Pix2Pix**, take an input image (in our case, the black-and-white 68-point landmark map) and learn to translate it into an output image (the photorealistic face).



To achieve this, the network consists of two competing models trained simultaneously:
1. **The Generator ($G$)**: Tries to draw a realistic face based on the landmark map.
2. **The Discriminator ($D$)**: Looks at a pair of images (Landmark Map + Face) and guesses if the face is a *Real* photo from the dataset or a *Fake* image drawn by the Generator.

---

## 🏗️ Deep Dive: Model Architecture & Layers

Since our preprocessed images are 256x256x3 pixels, the spatial relationship between the input (landmarks) and output (face) must be strictly preserved. Therefore, we use specialized architectures for both the Generator and Discriminator.

### 1. The Generator: U-Net Architecture
The Generator does not use a standard sequential CNN. Instead, it uses a **U-Net** architecture, which is an Encoder-Decoder network with **Skip Connections**.



#### **A. The Encoder (Downsampling)**
The Encoder acts as a feature extractor, compressing the 256x256 landmark map into a deep, abstract representation.
* **Layers**: It consists of a series of `Conv2d` layers with a stride of 2.
* **Activations**: `LeakyReLU` is used to prevent the "dying ReLU" problem during deep network training.
* **Normalization**: `BatchNorm2d` is applied to stabilize learning.
* **Progression**: As the spatial resolution halves (256 → 128 → 64 → ...), the number of feature channels doubles (64 → 128 → 256 → ...).

#### **B. The Bottleneck**
At the bottom of the "U", the image is compressed into a very dense tensor (e.g., 1x1x512). This forces the network to learn the most essential, high-level features of the desired expression.

#### **C. The Decoder (Upsampling) & Skip Connections**
The Decoder must reconstruct the 256x256 face from the bottleneck.
* **Layers**: It uses `ConvTranspose2d` (Deconvolution) layers with a stride of 2 to double the spatial resolution at each step.
* **Activations**: Standard `ReLU` is used here, followed by `BatchNorm2d`.
* **The "Magic" - Skip Connections**: If the network only used the bottleneck, it would lose the exact pixel locations of the eyes and mouth. To fix this, U-Net uses **skip connections**. The output of an Encoder layer is directly concatenated to the input of the corresponding Decoder layer. This feeds the exact spatial coordinates of the input landmarks directly to the output rendering layers.
* **Final Layer**: A `Tanh` activation function is used at the very end to scale the output pixels to the range `[-1, 1]`.

### 2. The Discriminator: PatchGAN Architecture
Instead of outputting a single "Real/Fake" probability for the entire image, we use a **PatchGAN** architecture. 



* **How it works**: The Discriminator is a standard Convolutional Neural Network, but it does not have fully connected (Dense) layers at the end. Instead, it outputs an $N \times N$ matrix (e.g., $30 \times 30$). 
* **Patch Evaluation**: Each value in this matrix represents the model's "Real/Fake" guess for a specific, overlapping *patch* of the image (e.g., a $70 \times 70$ pixel area). 
* **Why PatchGAN?**: It forces the Discriminator to focus on high-frequency details (textures, realistic skin, sharp edges) rather than the overall structure, which is already enforced by the Generator's L1 Loss.

---

## 🎯 The Training Objective (Loss Functions)

The network optimizes two primary loss functions to achieve photorealism:

1. **Adversarial Loss (BCE Loss)**: 
   The standard GAN game. The Generator tries to minimize this (fool the Discriminator), while the Discriminator tries to maximize it (catch the fakes).
   
2. **L1 Loss (Mean Absolute Error)**: 
   Adversarial loss alone can cause hallucinations. To ensure the generated face *exactly* matches the input landmark dots, we calculate the absolute pixel-by-pixel difference between the Generated Face and the Target Real Face. 

   $$\text{Total Loss} = \text{Adversarial Loss} + (\lambda \times \text{L1 Loss})$$
   
   *(Where $\lambda$ is usually set high, e.g., 100, to force structural adherence).*

---

## 🛠️ Data Preprocessing Pipeline
Before the model can train, the data must be perfectly formatted. This project automates the following process:
1. **Face Detection**: Uses `dlib.get_frontal_face_detector()` to crop out unnecessary background from the JAFFE dataset.
2. **Resizing**: Standardizes all crops to the required 256x256 resolution.
3. **Landmark Extraction**: Uses `dlib.shape_predictor()` to find 68 key structural coordinates.
4. **Conditioning Map Generation**: Draws white circles on a black matrix to create the 256x256 "dot skeletons."
5. **Formatting**: Converts grayscale outputs to 3-channel (BGR/RGB) tensors expected by the GAN's first Conv2D layer.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.11+
* PyTorch 2.2.2
* A Kaggle account (to run the notebook)

### Run on Kaggle
You can view the full code, including the preprocessing pipeline and the model architecture, directly on Kaggle. No local installation is required!

👉 **[View and run the Expression-DCGAN notebook here](https://www.kaggle.com/code/sepehrbayatnezhad/expression-dcgan)**

---
*Created for educational exploration into Conditional GAN architectures and facial morphology.*
