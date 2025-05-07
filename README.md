# 🎨 Realistic Neural Style Transfer

## 📌 Project Overview

This project proposes a refined neural style transfer (NST) architecture that improves upon traditional methods, particularly when applying abstract art styles to photographic content. Our approach emphasizes:

- Preserving the global structure and edges of the content image
- Transferring the high-level artistic tone and color distribution of the style image
- Reducing distortions common in patch-based or single-layer loss models

---

## 🚀 Motivation

Traditional NST often breaks down when abstract or minimalistic art styles are applied to detailed photographs. These limitations arise due to:
- Weak patch correspondence
- Oversimplified content loss
- Lack of structural guidance

We address these issues by introducing:
- Multi-layer content loss
- Edge-aware loss functions (Laplacian, Sobel)
- Total Variation (TV) loss for noise suppression

---

## ⚙️ Implementation

- **Baseline Models**:  
  - TensorFlow with pre-trained VGG19 (via TensorFlow Hub)  
  - PyTorch with custom loss functions and VGG19  
- **Hardware**: CPU and A100 GPU via Google Colab  
- **Data**: Publicly available photographs and artworks (e.g., Picasso)

---

## 📓 Notebook Walkthrough

We begin by loading a **Picasso painting** (style image) and a **skyline photo** (content image). Baseline outputs are generated using both Google's pre-trained NST model and an unmodified PyTorch VGG19 model.

### 🔍 Layer Inspection & Feature Engineering

- Visualized convolutional layers to determine their impact on stylization
- Created helper functions for:
  - Gram matrix computation
  - Feature extraction
  - Loss evaluations

### 🧠 Style and Content Layer Design

- **Content layer**: `conv_4` (from Gatys et al., 2016)
- **Style layers**: `conv_1` to `conv_5`

Initial experiments showed:
- Early layers (e.g., `conv_1`, `conv_2`) preserve structure
- Deeper layers (e.g., `conv_4`, `conv_5`) inject more stylistic features

---

## 🔧 Loss Functions

### 🔹 Multi-Layer Content Loss

Captures both low-level textures and high-level structure:

\\[
\\mathcal{L}_{\\text{content}}^{\\text{multi-layer}}(\\tilde{p}, \\tilde{x})
= \\sum_{l \\in L} w_l 
  \\left( 
    \\frac{1}{N_l M_l} 
    \\sum_{i=1}^{N_l} 
    \\sum_{j=1}^{M_l} 
      \\bigl(F_{ij}^l - P_{ij}^l\\bigr)^2 
  \\right)
\\]

### 🔹 Laplacian Edge Loss

Encourages edge sharpness using second-order derivatives:

\\[
L_{\\text{laplacian}}
= \\frac{1}{C\\,H\\,W}
\\sum_{c=1}^{C}\\sum_{h=1}^{H}\\sum_{w=1}^{W}
\\bigl((\\Delta\\tilde{p})_{c,h,w} - (\\Delta\\tilde{x})_{c,h,w}\\bigr)^{2}
\\]

### 🔹 Sobel Edge Loss (Proposed)

Improves over Laplacian by capturing directional gradients:

\\[
L_{\\text{sobel}} = MSE(Sₓ(p̂), Sₓ(x̂)) + MSE(Sᵧ(p̂), Sᵧ(x̂))
\\]

Empirical results showed \\(L_{\\text{sobel}}\\) outperformed \\(L_{\\text{laplacian}}\\) in preserving straight lines and contours, especially in architectural scenes.

### 🔹 Total Variation (TV) Loss

Suppresses high-frequency noise without blurring edges:

\\[
L_{\\text{tv}} = \\sum [ (x_{h,w+1} - x_{h,w})^2 + (x_{h+1,w} - x_{h,w})^2 ]
\\]

---

## 🔺 Final Total Loss Function

Our final formulation balances style, content, edge sharpness, and smoothness:

\\[
L_{\\text{total}} = \\alpha \\cdot L_{\\text{content}} + \\beta \\cdot L_{\\text{style}} + \\gamma \\cdot L_{\\text{sobel}} + \\delta \\cdot L_{\\text{tv}}
\\]

---

## 📈 Results

- Outperformed TensorFlow NST and ChatGPT-4o in structural preservation
- Better contour and edge clarity in abstract-to-photo transfers
- Cleaner and more coherent textures with reduced distortion

---

## 🧪 Challenges & Key Learnings

- Single-layer content loss fails to capture fine-grained structure
- Laplacian loss improves sharpness, but lacks directional awareness
- Sobel loss + TV loss significantly boost photorealism and reduce noise

---

## 📚 References

- Gatys et al. (2016), *A Neural Algorithm of Artistic Style*
- Johnson et al. (2016), *Perceptual Losses for Real-Time Style Transfer*
- Zhang et al. (2021), *Multi-layer Feature Fusion*
- Seif & Androutsos (2018), *Edge-Preserving Loss Functions*
- Reimann et al. (2022), *Structural Consistency in Style Transfer*

---

> For code and experiments, see the `TestingProposedMethods.ipynb` notebook.