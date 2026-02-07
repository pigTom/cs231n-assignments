# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Stanford CS231N Computer Vision course assignment collection. Three progressive assignments teach deep learning for computer vision:

- **Assignment 1**: Fundamentals (kNN, SVM, Softmax, two-layer neural networks)
- **Assignment 2**: Deep learning (CNNs, Batch Normalization, Dropout, PyTorch, RNN captioning)
- **Assignment 3**: Advanced topics (Vision Transformers, Diffusion Models, Self-Supervised Learning, CLIP/DINO)

## Development Commands

### Dataset Setup (Assignment 1)
```bash
cd assignment1/cs231n/datasets
bash get_datasets.sh
```

### Build Cython Extensions (Assignment 2)
```bash
cd assignment2/cs231n
python setup.py build_ext --inplace
```

### Install Dependencies
```bash
pip install -r assignment3/requirements.txt
```

### Run Jupyter Notebooks
```bash
jupyter notebook
```

## Architecture

Each assignment has the same structure:
- **Jupyter notebooks** (`.ipynb`) at the root provide instructions, tests, and visualizations
- **`cs231n/` package** contains the Python library students implement

### Core Library Structure (`cs231n/`)

| File | Purpose |
|------|---------|
| `layers.py` | Neural network layer forward/backward implementations |
| `layer_utils.py` | Convenience wrappers combining layers |
| `optim.py` | Optimization algorithms (SGD, Adam, RMSprop) |
| `solver.py` | Training loop orchestration |
| `gradient_check.py` | Numerical gradient verification |
| `data_utils.py` | Data loading and preprocessing |
| `classifiers/` | Complete model implementations |

### Implementation Pattern

Functions follow a forward/backward pattern with TODO blocks for student implementation:

```python
def layer_forward(x, params):
    """
    Forward pass computes output and cache.
    """
    out = None
    ###########################################################################
    # TODO: Implement the forward pass                                        #
    ###########################################################################
    # Student code here
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, params)
    return out, cache

def layer_backward(dout, cache):
    """
    Backward pass computes gradients from upstream derivative and cache.
    """
    # Similar TODO pattern
    return gradients
```

### Model API

Models must expose:
- `model.params`: dict mapping parameter names to numpy arrays
- `model.loss(X, y)`: returns (loss, gradients_dict)

### Solver Usage

```python
solver = Solver(model, data,
                update_rule='sgd',
                optim_config={'learning_rate': 1e-3},
                lr_decay=0.95,
                num_epochs=10,
                batch_size=100)
solver.train()
```

## Assignment-Specific Notes

### Assignment 2 Additions
- `fast_layers.py`: Optimized conv/pool using im2col
- `im2col_cython.pyx`: Cython-accelerated im2col (requires compilation)
- `coco_utils.py`: COCO dataset for image captioning
- `rnn_layers_pytorch.py`: PyTorch RNN implementations

### Assignment 3 Additions
- `transformer_layers.py`: Multi-head attention, positional encoding
- `simclr/`: Self-supervised contrastive learning (SimCLR model, contrastive loss)
- `gaussian_diffusion.py`, `unet.py`, `ddpm_trainer.py`: Diffusion model components
- `clip_dino.py`: Vision-language model utilities
