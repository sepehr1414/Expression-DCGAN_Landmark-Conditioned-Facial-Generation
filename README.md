# 🎭 Expression-DCGAN: Identity-Preserving Facial Expression Transfer

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/sepehrbayatnezhad/expression-dcgan)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C)
![dlib](https://img.shields.io/badge/dlib-Facial_Landmarks-green)

## 📖 Project Overview
This project implements an advanced **Conditional Generative Adversarial Network (cGAN)** engineered to synthesize realistic human facial expressions from structural facial landmark maps. 

Moving beyond standard Image-to-Image translation (like basic Pix2Pix), this architecture utilizes an **Encoder-Generator pipeline** combined with **Latent Space Optimization**. It successfully transfers a target emotion's geometric structure onto a subject while strictly preserving their original facial identity—verified dynamically using a pre-trained Facenet ResNet model.



---

## 🏗️ Deep Dive: Model Architecture

The core pipeline consists of three distinct deep learning networks working in tandem:

### 1. The Conditional Generator ($G$)
Unlike standard U-Nets that strictly map pixels to pixels, this Generator is a Latent-Conditioned DCGAN.
* **Inputs:** It takes a concatenated input during the forward pass: $G(z, y)$. 
  * $z$: A dense latent noise vector encoding abstract features (identity, lighting, skin texture).
  * $y$: A $256 \times 256$ spatial 68-point landmark map representing the geometric structural condition (the expression).
* **Mechanism:** It uses a series of upsampling Convolutional blocks to synthesize a photorealistic $256 \times 256 \times 3$ (RGB) face by interpreting the structural bounds of $y$ and rendering the textures encoded within $z$.

### 2. The Feature-Matching Discriminator ($D$)
This network goes far beyond standard binary classification to enforce high-frequency texture realism.
* **Input Concatenation:** It evaluates the spatial relationship between the face and the expression by concatenating them at the input level: `torch.cat([face_rgb, landmark_map], dim=1)`.
* **Dual Output:** The `forward` function returns `final_output` (the Real/Fake logit) **AND** a list of `feature_maps` containing the intermediate layer activations. 



### 3. The Inverse Encoder ($E$)
Standard GANs cannot map a real image back to its underlying latent vector. To solve this, a custom Encoder was built.
* **Architecture:** A deep CNN that downsamples the $256 \times 256$ RGB image back into the dense $z$ vector space ($128 \rightarrow 64 \rightarrow 32 \dots$).
* **Training Objective:** Trained *after* the Generator is frozen. It minimizes the Mean Squared Error (MSE) between its predicted $z$ and the true synthetic $z$ that was used by $G$ to create a batch of faces.

---

## 📉 The Advanced Loss Landscape
Generative models are notoriously unstable. This pipeline implements a robust, multi-objective loss landscape to prevent mode collapse and ensure structural fidelity:

1. **Adversarial Loss (BCE):** The standard minimax game between $G$ and $D$.
2. **Feature-Matching Loss (L1):** The Generator is heavily penalized based on the L1 distance between the Discriminator's *internal feature maps* of real faces versus generated faces. This forces $G$ to learn matching high-frequency textures (pores, hair) rather than just fooling the final classification layer.
3. **R1 Gradient Penalty Regularization:** A penalty applied to the real data during the Discriminator's training step to keep gradients smooth and prevent $D$ from overpowering $G$ too quickly.

---

## 🔬 Core Innovation: Expression Transfer via Latent Optimization

To transfer a new expression onto a real person's face *without* changing their fundamental identity, the code implements a custom Latent Space Optimization loop:

1. **Initialization:** The real Source Face is passed through the Encoder to extract an initial starting point: $z_{initial} = E(Source\_Face)$.
2. **Trainable Latent Vector:** The latent vector is detached from the graph and converted into a trainable parameter: `z.requires_grad_(True)`.
3. **Adam Optimization:** An independent Adam optimizer is instantiated to update the pixels of the noise vector $z$ over $N$ steps.
4. **Identity Preservation:** In the loop, $G$ renders a face conditioned on the *Target Expression Landmarks*. The code calculates an **Identity Loss** between this generated face and the original Source Face using a pre-trained **Facenet InceptionResnetV1**. 
5. **Convergence:** By backpropagating the Facenet loss directly into $z$, the model learns to shift the latent space to draw the exact same person, but the spatial condition forces the geometry to match the new expression.

---

## 📊 Automated Evaluation: FID & KID Search
The notebook contains an automated hyperparameter search loop. Instead of relying on subjective human visual inspection, the script iteratively evaluates different learning rates and regularization weights ($R1$, Feature-Matching $\lambda$). 

It mathematically scores the output using:
* **Fréchet Inception Distance (FID)**
* **Kernel Inception Distance (KID)**

These metrics utilize a pre-trained InceptionV3 network to compare the statistical distribution of the generated dataset against the real JAFFE dataset.

---

## 🚀 Run on Kaggle

You can view the full code, including the preprocessing pipeline, custom loss functions, and Latent Optimization loops directly on Kaggle. No local installation or GPU is required!

👉 **[View and run the Expression-DCGAN notebook here](https://www.kaggle.com/code/sepehrbayatnezhad/expression-dcgan)**
