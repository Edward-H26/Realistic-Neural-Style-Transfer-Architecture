# 🎨 CS445 Final Project: Realistic Neural Style Transfer

## 📌 Project Overview

This project proposes a refined neural style transfer (NST) architecture that improves upon traditional methods, particularly when applying abstract art styles to photographic content. Our approach emphasizes:

- Preserving the global structure and edges of the content image
- Transferring the high-level artistic tone and color distribution of the style image
- Reducing distortions common in patch-based or single-layer loss models

<div align="center">
  <img src="data/result.png" width="600" alt="Before and after style transfer"/>
</div>



## 🚀 Motivation

Traditional NST using style loss and content loss often breaks down when abstract or minimalistic art styles are applied to detailed photographs. These limitations arise due to:
- Weak patch correspondence
- Oversimplified content loss
- Lack of structural guidance

We address these issues by introducing:
- Multi-layer content loss
- Edge-aware loss functions (Laplacian, Sobel)
- Total Variation (TV) loss for noise suppression




| Component                           | Purpose                                       |
| ----------------------------------- | --------------------------------------------- |
| **Baseline VGG19** (TensorFlow Hub) | Produces initial stylisation and feature maps |
| **Multi‑Layer Content Loss**        | Matches high‑ and low‑level representations   |
| **Style Loss** (Gram matrices)      | Transfers global colour & texture             |
| **Sobel Edge Loss (ours)**          | Keeps straight edges & contours               |
| **Total Variation Loss**            | Removes high‑frequency noise                  |


---

## ⚙️ Implementation

- **Baseline Models**:  
  - TensorFlow with pre-trained VGG19 (via TensorFlow Hub)  
  - PyTorch with custom loss functions and VGG19  
- **Hardware**: CPU and A100 GPU via Google Colab  
- **Data**: Publicly available photographs and artworks (e.g., Picasso)

## ⚙️ Quick Start

1. **Clone & install dependencies**

   ```bash
   git clone https://github.com/HelenWu2004/CS445FinalProject.git
   cd CS445FinalProject
   python -m venv .venv && source .venv/bin/activate  # optional
   pip install -r requirements.txt
   ```
2. **Prepare data** – place your content images and style images in `data/`.
3. **Run the notebook**

   ```bash
   jupyter notebook notebooks/Realistic_NST.ipynb
   ```

4. **Adjust hyper‑parameters** (`alpha`, `beta`, `gamma`, `delta`) in the notebook or via CLI flags to fine‑tune the trade‑off between content fidelity and stylisation.

> **Tip:** The default weights `α:1  β:5  γ:1  δ:0.01` work well for most photographs. Increase `γ` for sharper edges or `δ` for smoother backgrounds.


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
### 🔧 Multi-Layer Content Loss

To capture a richer hierarchy of features, we implemented **multi-layer content loss**:

$$
\mathcal{L}_{\text{content}}^{\text{multi-layer}}(\tilde{p}, \tilde{x}) =
\sum_{l \in L} w_l \left(
\frac{1}{N_l M_l} \sum_{i=1}^{N_l} \sum_{j=1}^{M_l}
(F_{ij}^l - P_{ij}^l)^2
\right)
$$



### 🌀 Laplacian Edge Loss

To address distortion from abstract styles, we added Laplacian edge loss using the kernel:

```
[ 0, -1,  0]
[-1,  4, -1]
[ 0, -1,  0]
```

This sharpens edges by penalizing differences in second derivatives:

$$
L_{\text{laplacian}} =
\frac{1}{C\,H\,W}
\sum_{c,h,w} \left((\Delta\tilde{p})_{c,h,w} - (\Delta\tilde{x})_{c,h,w}\right)^2
$$
Combined with content loss:

$$
L_{\text{content}} = \sum_l w_l \cdot \text{MSE}(F^l(\tilde{p}), F^l(\tilde{x})) + L_{\text{laplacian}}
$$

### ➤ Sobel Edge Loss (Proposed)

Empirical results showed that **Sobel edge loss** better preserved edge directionality and contours:

$$
L_{\text{sobel}} = MSE(Sₓ(\tilde{p}), Sₓ(\tilde{x})) + MSE(Sᵧ(\tilde{p}), Sᵧ(\tilde{x}))
$$

### 🌫️ Total Variation (TV) Loss

We also observed noise in uniform regions (e.g., sky). **TV loss** was added to reduce high-frequency artifacts:

$$
L_{\text{tv}} = \sum [ (x_{h,w+1} - x_{h,w})^2 + (x_{h+1,w} - x_{h,w})^2 ]
$$

Full equation:

$$L_{\text{tv}} =
\frac{1}{BCHW} \sum_{b,c,h,w}
\left[(\tilde{x}_{b,c,h,w+1} - \tilde{x}_{b,c,h,w})^2 + (\tilde{x}_{b,c,h+1,w} - \tilde{x}_{b,c,h,w})^2\right]
$$


### 🧮 Final Loss Function

Our final total loss combines all components for robust stylization:

$$L_{\text{total}} =
\alpha \cdot L_{\text{content}} +
\beta \cdot L_{\text{style}} +
\gamma \cdot L_{\text{sobel}} +
\delta \cdot L_{\text{tv}}
$$


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
