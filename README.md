# DAMCC 🚀
### (A Deep Autoregressive Model for Dynamic Combinatorial Complexes)

📄 **Read the original paper:** [arXiv Link](https://arxiv.org/abs/2503.01999)

---

## 🛠️ Dependencies

> **Recommended Python version:** `3.11.3`

Set up your environment (using [conda](https://docs.conda.io/en/latest/), or your preferred tool):

```bash
conda create -n damcc python=3.11.3
conda activate damcc
```

### 1️⃣ Install TopoModelX

`TopoModelX` is available on PyPI:

```bash
pip install topomodelx
```

### 2️⃣ Install PyTorch & Related Packages

Install `torch`, `torch-scatter`, and `torch-sparse` (with or without CUDA):

```bash
pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/${CUDA}
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html
# Optional: torch-cluster (recommended by TopoModelX, but not required for DAMCC)
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
```

> Replace `${CUDA}` with `cpu`, `cu102`, `cu113`, or `cu115` according to your setup (`torch.version.cuda`).

### 3️⃣ Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

---

## 📓 Quick Start

Check out the [tutorial notebook](notebooks/damcc_tutorial.ipynb) for a hands-on introduction.

> ⚠️ **Note:** Some users have reported issues on Apple M-series chips.

---

Happy coding! 💻✨
