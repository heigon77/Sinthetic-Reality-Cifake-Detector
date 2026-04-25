# 🖼️ Real vs. AI-Generated Image Classifier

A deep learning project that classifies images as either **real photographs** or **AI-generated images** (produced by Stable Diffusion), using a custom CNN trained on the **CIFAKE** dataset. The notebook also includes **GradCAM** visualizations to interpret which regions of an image the model focuses on when making predictions.

---

## 📓 Notebook Overview — `projeto.ipynb`

The notebook is organized into three main sections:

### 1. Dataset Preparation
- Loads the **CIFAKE** dataset, which combines the original CIFAR-10 images with 60,000 Stable Diffusion-generated counterparts
- Splits data into **90% training / 10% validation** sets
- Applies standard preprocessing: resize to `32×32`, `ToTensor`, and ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- A custom `ImageDataset` class handles label parsing from filenames (`real_*` → 0, `fake_*` → 1)

### 2. CNN Training & Evaluation
- A custom **HandCraftNet** architecture is defined with:
  - Two convolutional blocks (`Conv2d → BatchNorm → ReLU → MaxPool`)
  - Skip-connection-style feature concatenation between both blocks
  - A fully connected classifier with Dropout (0.20)
  - He weight initialization
- Trained with **Adam** optimizer (`lr=1e-5`) and a **StepLR** scheduler
- Loss: **CrossEntropyLoss** + L2 regularization
- Evaluated using **accuracy**, **per-class accuracy**, and **Cohen's Kappa score**
- Training/validation curves are plotted and the best model is saved as `modelHF.pth`

### 3. GradCAM Visualization
- Implements a custom **Gradient-weighted Class Activation Map (GradCAM)** to highlight discriminative image regions
- Generates side-by-side comparisons of the original image and its activation heatmap
- Runs inference on two custom generated image sets (`Generated/in_test` and `Generated/out_test`) to assess generalization beyond the CIFAKE dataset
- Results are saved to the `GradCAM/` directory

---

## 📁 Project Structure

```
.
├── projeto.ipynb              # Main notebook
├── Dataset/
│   ├── train_all/             # Training images (real_*.jpg / fake_*.jpg)
│   └── test_all/              # Test images
├── Generated/
│   ├── in_test/               # Custom generated images (in-distribution)
│   ├── out_test/              # Custom generated images (out-of-distribution)
│   └── GradCAM/
│       ├── in_test/           # GradCAM results for in_test
│       └── out_test/          # GradCAM results for out_test
├── GradCAM/                   # GradCAM outputs from the test set
└── Model/
    └── modelHF.pth            # Saved model weights
```

---

## 🛠️ Tech Stack

| Component | Library |
|---|---|
| Deep Learning | PyTorch |
| Dataset Handling | `torch.utils.data`, `torchvision` |
| Image Processing | Pillow, OpenCV (`cv2`) |
| Training Logging | `torch_snippets` (Report) |
| Evaluation Metrics | `sklearn` (accuracy, Cohen's Kappa) |
| Visualization | Matplotlib |

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9+**
- NVIDIA GPU with CUDA (CPU fallback is supported but slow)

### Install dependencies

```bash
pip install torch torchvision pillow opencv-python scikit-learn matplotlib torch_snippets torchsummary
```

### Dataset

Download the **CIFAKE** dataset and place it under the `Dataset/` directory, following the structure above. Image filenames must follow the pattern `real_*.jpg` or `fake_*.jpg` for the label parser to work correctly.

### Running

Open the notebook and run all cells in order:

```bash
jupyter notebook projeto.ipynb
```

---

## 📊 Model Architecture

```
HandCraftNet
├── conv1: Conv2d(3→32) → BatchNorm → ReLU → MaxPool
├── conv2: Conv2d(32→256) → BatchNorm → ReLU → MaxPool
├── [interpolate conv1 output + concat with conv2 output]
└── classifier: Linear(18432→64) → ReLU → Dropout(0.2) → Linear(64→2)
```

---

## 🔬 GradCAM

The GradCAM implementation uses the gradients flowing back through the convolutional layers to weight the feature maps, producing a heatmap that highlights the regions most relevant to the model's prediction. Results are saved as side-by-side JPEGs (`original | heatmap`).

---

## 📄 License

This project is intended for academic and research purposes.
