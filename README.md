# 🌊 SwiftCFD — Neural Network Flow Field Predictor

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Space-SwiftCFD-blue)](https://huggingface.co/spaces/vamsigudipati/SwiftCFD)
[![Model on HF](https://img.shields.io/badge/🤗%20Model-deepcfd--model-orange)](https://huggingface.co/vamsigudipati/deepcfd-model)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SwiftCFD is a deep learning surrogate model for Computational Fluid Dynamics (CFD).
It predicts steady-state laminar flow fields (Ux, Uy, pressure) around 2D obstacles
in **~50ms** — replacing hours of traditional CFD simulation.

---

## 🚀 Live Demo

👉 **[Try SwiftCFD on Hugging Face Spaces](https://huggingface.co/spaces/vamsigudipati/SwiftCFD)**

---

## 📦 Resources

| Resource | Link |
|---|---|
| 🤗 Model weights | [vamsigudipati/deepcfd-model](https://huggingface.co/vamsigudipati/deepcfd-model) |
| 🌊 Live Demo | [SwiftCFD Space](https://huggingface.co/spaces/vamsigudipati/SwiftCFD) |
| 📊 Dataset | [DeepCFD — Zenodo](https://zenodo.org/record/3666056) |
| 📄 Original Paper | [DeepCFD — arXiv](https://arxiv.org/abs/2004.08826) |

---

## 🏗️ Architecture

- **Model:** UNetEx (U-Net with skip connections + max unpooling)
- **Input:** 3-channel tensor `[SDF, Ux_inlet, Uy_inlet]` — shape `(3, 172, 79)`
- **Output:** 3-channel tensor `[Ux, Uy, p]` — shape `(3, 172, 79)`
- **Filters:** `[8, 16, 32, 32]` | **Kernel size:** `5×5`
- **Parameters:** ~500K | **Model size:** ~3.3MB

---

## 📊 Training

| | Pass 1 | Pass 2 (Fine-tune) |
|---|---|---|
| Script | `train_pass1.py` | `train_finetune.py` |
| Optimizer | Adam | AdamW (weight_decay=0.02) |
| Learning rate | 1e-3 | 1e-4 |
| Batch size | 64 | 64 |
| Scheduler | ReduceLROnPlateau (patience=20) | ReduceLROnPlateau (patience=50) |
| Early stopping | patience=100 | patience=200 |
| Max epochs | 2000 | 2000 |
| Best Val MSE | — | **0.739** |

---

## 📈 Results

| Field | R² | Slope |
|---|---|---|
| Ux | **0.9974** | 0.9977 |
| Uy | **0.9960** | 0.9983 |
| p  | **0.7552** | 0.7647 |

---

## 🛠️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/vamsigudipati/SwiftCFD.git
cd SwiftCFD
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
```bash
wget https://zenodo.org/record/3666056/files/DeepCFD.zip
unzip DeepCFD.zip
# → dataX.pkl and dataY.pkl
```

---

## 🏋️ Training

### Pass 1 — Train from scratch
```bash
python train_pass1.py \
  --dataX dataX.pkl \
  --dataY dataY.pkl \
  --output checkpoint.pt
```

### Pass 2 — Fine-tune
```bash
python train_finetune.py \
  --checkpoint checkpoint.pt \
  --dataX dataX.pkl \
  --dataY dataY.pkl \
  --output mymodel_v2.pt
```

---

## 📊 Visualizations

```bash
# Plot sample predictions + scatter + training curves
python visualize.py \
  --checkpoint mymodel_v2.pt \
  --dataX dataX.pkl \
  --dataY dataY.pkl \
  --samples 103 487 354 563 520 \
  --out_dir ./assets

# Pass 1 vs Pass 2 comparison
python visualize.py \
  --checkpoint mymodel_v2.pt \
  --checkpoint1 checkpoint.pt \
  --dataX dataX.pkl \
  --dataY dataY.pkl \
  --plot comparison
```

---

## 🖥️ Run the App Locally

```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```
SwiftCFD/
├── app.py                  ← Streamlit demo app
├── train_pass1.py          ← Pass 1: train from scratch
├── train_finetune.py       ← Pass 2: fine-tuning
├── visualize.py            ← Visualization scripts
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── DOCUMENTATION.md        ← Full project documentation
├── LICENSE                 ← MIT License
├── .gitignore
└── assets/                 ← Result plots and snapshots
    ├── pass1_sample0.png
    ├── pass1_sample1.png
    ├── pass1_sample2.png
    ├── training_curves_v2.png
    ├── finetune_sample103.png
    ├── finetune_sample487.png
    ├── finetune_sample354.png
    ├── finetune_sample563.png
    ├── finetune_sample520.png
    └── scatter.png
```

> **Note:** Model weights (`*.pt`) and dataset (`*.pkl`) are NOT stored
> in this repository. They are hosted on
> [Hugging Face](https://huggingface.co/vamsigudipati/deepcfd-model)
> and downloaded automatically at runtime.

---

## ⚠️ Known Limitations

- Best results when obstacle is placed in the **left-center third** of the
  domain (x = 30–80), matching the training data distribution
- Fixed input resolution of `172 × 79` cells
- Laminar flow only (low Reynolds number)
- 2D predictions only

See [DOCUMENTATION.md](DOCUMENTATION.md) for full details.

---

## 🙏 Acknowledgements

Based on [DeepCFD](https://github.com/mdribeiro/DeepCFD) by Ribeiro et al.

```bibtex
@article{deepcfd2020,
  title   = {DeepCFD: Efficient Steady-State Laminar Flow Approximation
             with Deep Convolutional Neural Networks},
  author  = {Ribeiro, Mateus D. and Rehman, Auwal and Ahmed, Shahood
             and Dengel, Andreas},
  journal = {arXiv preprint arXiv:2004.08826},
  year    = {2020}
}
```