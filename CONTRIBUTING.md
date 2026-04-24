# Contributing to SwiftCFD 🌊

Thanks for your interest in SwiftCFD! Whether you want to fix a bug, suggest a feature, improve the docs, or just try things out — you are very welcome here. This project is a one-person side project, so any contribution, however small, genuinely makes a difference.

---

## Ways to Contribute

| What | How |
|---|---|
| 🐛 **Bug report** | [Open an issue](https://github.com/vamsigudipati/SwiftCFD/issues/new) with steps to reproduce |
| 💡 **Feature idea** | Open a discussion or issue describing the idea and its motivation |
| 🔧 **Pull request** | Fork → branch → implement → open a PR against `main` |
| 🧪 **Testing** | Run the training scripts or Streamlit app and report anything unexpected |
| 📝 **Documentation** | Typo fixes, clarifications, or new examples are always appreciated |

---

## Setting Up the Dev Environment

### 1. Clone the repo

```bash
git clone https://github.com/vamsigudipati/SwiftCFD.git
cd SwiftCFD
```

### 2. Install dependencies

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

The training dataset is hosted on **Zenodo** (the original DeepCFD dataset):

```
https://zenodo.org/record/3666056
```

Download and unzip `DeepCFD.zip` into the project root so that `dataX.pkl` and `dataY.pkl` are available.

### 4. (Optional) Get the pretrained weights

Model weights are hosted on Hugging Face Hub and are downloaded automatically by the Streamlit app at runtime. For local training you do not need them — training starts from scratch in Pass 1 and fine-tunes in Pass 2.

---

## Training Scripts

SwiftCFD uses a **two-pass training strategy**:

| Script | Purpose |
|---|---|
| `train_pass1.py` | Train the UNetEx model from scratch on the full dataset. Produces a `checkpoint.pt` baseline. |
| `train_finetune.py` | Fine-tune from the Pass 1 checkpoint with refined hyperparameters (AdamW, lower LR, longer patience). Produces the final model weights. |

Run them in order:

```bash
python train_pass1.py
python train_finetune.py
```

Both scripts save checkpoints to disk and print validation metrics (MSE, R²) at each epoch.

---

## Live Demo

You can try SwiftCFD without any local setup:

👉 **[https://huggingface.co/spaces/vamsigudipati/SwiftCFD](https://huggingface.co/spaces/vamsigudipati/SwiftCFD)**

---

## Model Weights

The fine-tuned model weights are publicly available on Hugging Face Hub:

👉 **[https://huggingface.co/vamsigudipati/deepcfd-model](https://huggingface.co/vamsigudipati/deepcfd-model)**

---

## Background Reading

SwiftCFD is built on the **DeepCFD** architecture by Ribeiro et al. (2020). If you want to understand the UNetEx model and the training dataset in depth, the original paper is a great starting point:

> Ribeiro, M. D., Rehman, A., Ahmed, S., & Dengel, A. (2020).
> **DeepCFD: Efficient Steady-State Laminar Flow Approximation with Deep Convolutional Neural Networks.**
> arXiv:2004.08826. https://arxiv.org/abs/2004.08826

---

## Future Improvement Ideas

These are areas called out in [`DOCUMENTATION.md`](DOCUMENTATION.md) that would make great contributions:

- **Horizontal flip augmentation** — double effective dataset size and improve symmetry generalisation
- **Physics-informed loss** — add a continuity equation penalty (∇·u = 0) to the loss function
- **Turbulent flow extension** — extend to RANS/turbulent regimes beyond the current laminar scope
- **3D UNet extension** — adapt the architecture to predict 3D flow fields
- **Multi-obstacle support** — handle domains with more than one obstacle
- **Reynolds number as input channel** — condition the model on Re to generalise across flow regimes

If you pick up any of these, feel free to open a draft PR early — happy to give feedback along the way.

---

## Code Style

- Follow existing code conventions in the scripts (PEP 8, descriptive variable names)
- Keep PRs focused — one logical change per PR makes review much easier

---

Thanks again for considering a contribution! 🚀
