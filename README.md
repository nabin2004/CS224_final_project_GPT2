# WheelSonnet: GPT-2 From Scratch — Sentiment, Paraphrase Detection & Sonnet Generation

<p align="center">
  <img src="https://img.shields.io/badge/Model-GPT--2%20355M-blue" alt="Model"/>
  <img src="https://img.shields.io/badge/Framework-PyTorch-red" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/ONNX-Exported-green" alt="ONNX"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-orange" alt="License"/>
  <img src="https://img.shields.io/badge/HuggingFace-nabin2004-yellow" alt="HuggingFace"/>
</p>

> **Stanford CS 224N — Default Final Project: Build GPT-2**
>
> A from-scratch implementation of the GPT-2 architecture (124M & 355M), fine-tuned for **sentiment classification** (SST-5 & CFIMDB), **paraphrase detection** (Quora), and **Shakespearean sonnet generation** — with an additional **instruction-tuning** stage and **ONNX export** for efficient deployment.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Released Models on Hugging Face](#released-models-on-hugging-face)
- [Project Structure](#project-structure)
- [Setup & Reproducibility](#setup--reproducibility)
  - [Prerequisites](#prerequisites)
  - [Environment Setup](#environment-setup)
  - [Hardware Requirements](#hardware-requirements)
- [Part 1 — GPT-2 Core Implementation](#part-1--gpt-2-core-implementation)
  - [Causal Self-Attention](#causal-self-attention)
  - [GPT-2 Transformer Layer (Pre-LN)](#gpt-2-transformer-layer-pre-ln)
  - [GPT-2 Model](#gpt-2-model)
  - [AdamW Optimizer](#adamw-optimizer)
  - [Running Sanity Checks](#running-sanity-checks)
- [Part 2 — Downstream Tasks](#part-2--downstream-tasks)
  - [Task A: Sentiment Classification (SST-5 & CFIMDB)](#task-a-sentiment-classification-sst-5--cfimdb)
  - [Task B: Paraphrase Detection (Quora)](#task-b-paraphrase-detection-quora)
  - [Task C: Sonnet Generation](#task-c-sonnet-generation)
- [Part 3 — Extensions](#part-3--extensions)
  - [Instruction Tuning](#instruction-tuning)
  - [ONNX Export & Quantization](#onnx-export--quantization)
- [Reproducing Results](#reproducing-results)
- [Using the Released Models](#using-the-released-models)
  - [Download from Hugging Face](#download-from-hugging-face)
  - [Loading a Checkpoint for Inference](#loading-a-checkpoint-for-inference)
  - [Running ONNX Inference](#running-onnx-inference)
- [Evaluation Metrics](#evaluation-metrics)
- [Citation & Acknowledgements](#citation--acknowledgements)
- [License](#license)

---

## Overview

This project implements the full GPT-2 architecture from scratch (without using `nn.Transformer` or HuggingFace's model forward pass), loads pre-trained weights from OpenAI's GPT-2 checkpoints, and fine-tunes on three downstream NLP tasks:

| Task | Dataset | Method | Metric |
|------|---------|--------|--------|
| **Sentiment Classification** | SST-5 (5-class) / CFIMDB | Last-token cloze-style classification | Accuracy, Macro-F1 |
| **Paraphrase Detection** | Quora Question Pairs | Cloze-style binary classification | Accuracy, Macro-F1 |
| **Sonnet Generation** | Shakespeare Sonnets | Autoregressive language modeling (cross-entropy) | chrF |

The best performing sonnet model (`gpt2-medium`, 355M parameters) was further **instruction-tuned** for poetic Q&A and **exported to ONNX** (with INT8 quantization) for lightweight deployment.

---

## Architecture

The implementation follows the **Pre-LayerNorm GPT-2** variant:

```
Input Tokens
    │
    ▼
┌──────────────────┐
│ Token Embedding   │  (nn.Embedding, vocab_size × d)
│ + Position Embed  │  (nn.Embedding, max_seq_len × d)
│ + Dropout         │
└────────┬─────────┘
         │
    ┌────▼────┐ × L layers
    │ LayerNorm       │
    │ Causal MHA      │  (Q, K, V projections → scaled dot-product → causal mask)
    │ + Residual       │
    │ LayerNorm       │
    │ FFN (GELU)      │  (d → 3d → d)
    │ + Residual       │
    └────┬────┘
         │
    ┌────▼────┐
    │ Final LayerNorm │
    │ Task Head       │  (Classification / LM head)
    └─────────────────┘
```

**Key design choices:**
- **Pre-LayerNorm** (LN before attention and FFN, not after) — matches GPT-2's actual architecture
- **Causal mask** via `torch.tril` applied in attention scores before softmax
- **Weight tying** between input word embeddings and the LM output projection (`hidden_state @ E^T`)
- **Top-p (nucleus) sampling** with temperature scaling for generation

---

## Released Models on Hugging Face

All trained checkpoints are publicly available on Hugging Face:

| Model | HF Repo | Size | Description | Updated |
|-------|---------|------|-------------|---------|
| **WheelSonnet2-355M** | [`nabin2004/WheelSonnet2-355M`](https://huggingface.co/nabin2004/WheelSonnet2-355M) | 4.88 GB | GPT-2 Medium fine-tuned on Shakespeare sonnets (100 epochs, lr=1e-5) | Aug 4, 2025 |
| **WheelSonnet2-355M-it** | [`nabin2004/WheelSonnet2-355M-it`](https://huggingface.co/nabin2004/WheelSonnet2-355M-it) | 4.88 GB | Instruction-tuned variant for poetic Q&A (70 epochs on top of base) | Aug 6, 2025 |
| **WheenSonnet-355M-onnx** | [`nabin2004/WheenSonnet-355M-onnx`](https://huggingface.co/nabin2004/WheenSonnet-355M-onnx) | 652.5 MB | ONNX-exported sonnet model (FP32) | Aug 7, 2025 |
| **WheenSonnet-355M-quant-onnx** | [`nabin2004/WheenSonnet-355M-quant-onnx`](https://huggingface.co/nabin2004/WheenSonnet-355M-quant-onnx) | 164.0 MB | INT8 dynamically quantized ONNX model (**~30× smaller** than PyTorch) | Aug 7, 2025 |

> The ONNX quantized model achieves a **30× compression ratio** (4.88 GB → 164 MB) compared to the original PyTorch checkpoint, enabling CPU-only inference.

### Model Lineage

```
OpenAI GPT-2 Medium (355M pre-trained weights)
    │
    ▼
WheelSonnet2-355M              ← Fine-tuned on Shakespeare sonnets (100 epochs)
    │
    ├──▶ WheelSonnet2-355M-it  ← Instruction-tuned on poetic Q&A (70 epochs)
    │
    ├──▶ WheenSonnet-355M-onnx ← ONNX export (FP32, 652 MB)
    │
    └──▶ WheenSonnet-355M-quant-onnx ← INT8 quantized ONNX (164 MB)
```

---

## Project Structure

```
.
├── models/
│   ├── __init__.py
│   ├── base_gpt.py              # GPT pre-trained model base class (weight init, loading)
│   └── gpt2.py                  # GPT-2 model: embed → encode → forward, weight loading from HF
├── modules/
│   ├── attention.py             # Causal multi-head self-attention (from scratch)
│   └── gpt2_layer.py            # Single GPT-2 transformer block (Pre-LN, GELU FFN)
├── config.py                    # PretrainedConfig & GPT2Config (hyperparameters & architecture)
├── utils.py                     # Utility functions (extended attention mask, caching, etc.)
├── optimizer.py                 # Custom AdamW optimizer implementation
├── classifier.py                # Sentiment classification (SST-5, CFIMDB) training & eval
├── paraphrase_detection.py      # Paraphrase detection (Quora) training & eval
├── sonnet_generation.py         # Sonnet generation: SonnetGPT model, training, generation
├── instruct_tune.py             # Instruction-tuning pipeline for poetic Q&A
├── datasets.py                  # Dataset classes (ParaphraseDetection, Sonnets)
├── evaluation.py                # Evaluation: accuracy, F1, chrF for sonnets
├── optimizer_test.py            # Unit tests for AdamW implementation
├── sanity_check.py              # Sanity checks for GPT-2 model components
├── check_train.py               # Training verification utilities
├── prepare_submit.py            # Submission file preparation
├── env.yml                      # Conda environment specification
├── setup.sh                     # One-command environment setup
├── data/
│   ├── ids-sst-{train,dev,test-student}.csv    # SST-5 sentiment dataset
│   ├── ids-cfimdb-{train,dev,test-student}.csv  # CFIMDB sentiment dataset
│   ├── quora-{train,dev,test-student}.csv       # Quora paraphrase dataset
│   ├── sonnets.txt                              # Shakespeare sonnets (training)
│   ├── sonnets_held_out.txt                     # Held-out sonnets (first 3 lines only)
│   ├── sonnets_held_out_dev.txt                 # Dev sonnets for generation eval
│   ├── TRUE_sonnets_held_out_dev.txt            # Ground truth for chrF scoring
│   └── 001_first_1000.jsonl                     # Instruction-tuning data (poetic Q&A)
├── predictions/                                 # Output predictions directory
└── LICENSE                                      # Apache License 2.0
```

---

## Setup & Reproducibility

### Prerequisites

- **Python**: 3.8+
- **CUDA**: 11.x+ (recommended for GPU training)
- **Conda**: Miniconda or Anaconda
- **Git LFS**: Required for downloading HF model checkpoints
- **~12 GB GPU VRAM** for `gpt2-medium` training (e.g., NVIDIA RTX 3080/3090, A100, etc.)

### Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/CS224_final_project_GPT2.git
cd CS224_final_project_GPT2

# 2. Create and activate the conda environment
conda env create -f env.yml
conda activate cs224n_dfp

# 3. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

<details>
<summary><b>Full dependency list (env.yml)</b></summary>

```yaml
name: cs224n_dfp
channels:
  - defaults
dependencies:
  - python=3.8
  - pip
  - pip:
      - torch
      - torchvision
      - torchaudio
      - tqdm==4.58.0
      - requests==2.25.1
      - importlib-metadata==3.7.0
      - filelock==3.0.12
      - sklearn==0.0
      - tokenizers==0.20
      - explainaboard_client==0.0.7
      - einops==0.8.0
      - transformers==4.46.3
      - sacrebleu==2.5.1
```

</details>

### Hardware Requirements

| Task | Model Size | Min GPU VRAM | Approx. Training Time |
|------|-----------|-------------|----------------------|
| Sentiment (SST-5) | `gpt2` (124M) | 4 GB | ~30 min (10 epochs) |
| Paraphrase Detection | `gpt2` (124M) | 6 GB | ~1 hr (10 epochs) |
| Sonnet Generation | `gpt2-medium` (355M) | 12 GB | ~2–3 hrs (100 epochs) |
| Instruction Tuning | `gpt2-medium` (355M) | 12 GB | ~4–6 hrs (100 epochs) |

---

## Part 1 — GPT-2 Core Implementation

### Causal Self-Attention

**File:** [`modules/attention.py`](modules/attention.py)

Implements scaled dot-product attention with a causal (lower-triangular) mask:

```
Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) + M_causal ) V
```

where `M_causal` is `-inf` for positions above the diagonal (future tokens).

Key implementation details:
- Q, K, V projections via separate `nn.Linear` layers
- Head splitting using `einops.rearrange`: `'b t (h d) -> b h t d'`
- Causal mask via `torch.tril` applied before softmax
- Attention dropout applied to the attention probability matrix

### GPT-2 Transformer Layer (Pre-LN)

**File:** [`modules/gpt2_layer.py`](modules/gpt2_layer.py)

Each layer follows the Pre-LayerNorm pattern:

```
x → LayerNorm → CausalSelfAttention → Dropout → + x (residual)
  → LayerNorm → FFN(GELU)           → Dropout → + x (residual)
```

The FFN expands from `d` to `3d` (intermediate size) with GELU activation, then projects back to `d`.

### GPT-2 Model

**File:** [`models/gpt2.py`](models/gpt2.py)

- **Embedding:** Token embeddings + learned positional embeddings + dropout
- **Encoding:** Stack of `L` GPT-2 layers with extended attention mask
- **Output:** Final LayerNorm → returns both `last_hidden_state` (all tokens) and `last_token` (for classification)
- **Weight tying:** The `hidden_state_to_token` method computes `hidden_state @ word_embedding.weight.T`
- **Pre-trained weight loading:** `from_pretrained()` remaps OpenAI GPT-2 weights (Conv1D → Linear) from HuggingFace

### AdamW Optimizer

**File:** [`optimizer.py`](optimizer.py)

Custom implementation of AdamW with:
1. Exponential moving averages for first and second moments (m_t, v_t)
2. Bias correction (efficient version from [Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980))
3. **Decoupled weight decay** applied after the gradient update step (as per [Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101))

### Running Sanity Checks

```bash
# Test the AdamW optimizer implementation
python optimizer_test.py

# Test the GPT-2 model components (attention, layers, forward pass)
python sanity_check.py
```

---

## Part 2 — Downstream Tasks

### Task A: Sentiment Classification (SST-5 & CFIMDB)

**File:** [`classifier.py`](classifier.py)

Uses the **last-token hidden state** from GPT-2 as the sentence representation, passed through a classification head for 5-class (SST) or 2-class (CFIMDB) sentiment prediction.

**Training modes:**
- `last-linear-layer`: Freeze all GPT-2 parameters; train only the classification head
- `full-model`: Fine-tune all parameters end-to-end

```bash
# Train on SST-5 and CFIMDB (last-linear-layer mode)
python classifier.py --use_gpu --epochs 10 --lr 1e-3 --fine-tune-mode last-linear-layer

# Train with full fine-tuning
python classifier.py --use_gpu --epochs 10 --lr 1e-5 --fine-tune-mode full-model
```

### Task B: Paraphrase Detection (Quora)

**File:** [`paraphrase_detection.py`](paraphrase_detection.py)

Frames paraphrase detection as a **cloze-style classification** task:

```
Question 1: "{sentence1}"
Question 2: "{sentence2}"
Are these questions asking the same thing?
```

The model predicts whether the next token should be "yes" (token 8505) or "no" (token 3919).

```bash
# Train paraphrase detection
python paraphrase_detection.py --use_gpu --epochs 10 --lr 1e-5 --model_size gpt2

# With gpt2-medium (355M)
python paraphrase_detection.py --use_gpu --epochs 10 --lr 1e-5 --model_size gpt2-medium
```

### Task C: Sonnet Generation

**File:** [`sonnet_generation.py`](sonnet_generation.py)

Trains GPT-2 as an autoregressive language model on Shakespeare's sonnets. Uses **teacher forcing** with cross-entropy loss (shifted logits vs. shifted labels). Generation uses **top-p (nucleus) sampling** with temperature.

```bash
# Train sonnet generation with gpt2 (124M)
python sonnet_generation.py --use_gpu --epochs 10 --lr 1e-5 --model_size gpt2

# Train with gpt2-medium (355M) — this produces the WheelSonnet2-355M model
python sonnet_generation.py --use_gpu --epochs 100 --lr 1e-5 --model_size gpt2-medium \
    --temperature 0.7 --top_p 0.9
```

**Generation parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--temperature` | 1.2 | Softmax temperature (lower = more deterministic) |
| `--top_p` | 0.9 | Nucleus sampling cumulative probability cutoff |

---

## Part 3 — Extensions

### Instruction Tuning

**File:** [`instruct_tune.py`](instruct_tune.py)

Takes the pre-trained sonnet model and fine-tunes it on a conversational instruction dataset (`data/001_first_1000.jsonl`) formatted as:

```
User: Write a sonnet about the moon.
Assistant: [Generated sonnet]<|endoftext|>
```

This enables the model to follow natural language instructions for poetry generation.

```bash
# Instruction-tune from the pre-trained sonnet checkpoint
python instruct_tune.py --use_gpu \
    --model_path <path-to-sonnet-checkpoint>.pt \
    --train_path data/001_first_1000.jsonl \
    --epochs 100 --lr 1e-5 --batch_size 4 \
    --model_size gpt2-medium \
    --temperature 0.85 --top_p 0.9
```

### ONNX Export & Quantization

The trained PyTorch model was exported to ONNX format for cross-platform inference, and further quantized to INT8 for efficient CPU deployment:

| Format | Size | Compression | Inference Runtime |
|--------|------|-------------|-------------------|
| PyTorch (`.pt`) | 4.88 GB | — | PyTorch + CUDA |
| ONNX (FP32) | 652.5 MB | ~7.5× | ONNX Runtime |
| ONNX (INT8 quantized) | 164.0 MB | ~30× | ONNX Runtime (CPU) |

```python
# Example ONNX export (after training)
import torch

model = SonnetGPT(args)
model.load_state_dict(saved['model'])
model.eval()

dummy_input_ids = torch.randint(0, 50257, (1, 128))
dummy_mask = torch.ones(1, 128, dtype=torch.long)

torch.onnx.export(
    model, (dummy_input_ids, dummy_mask),
    "sonnet_gpt2.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {1: "seq_len"},
        "attention_mask": {1: "seq_len"},
        "logits": {1: "seq_len"}
    },
    opset_version=14
)
```

---

## Reproducing Results

Below is the complete sequence to reproduce all results from scratch:

```bash
# ── Step 0: Environment ──────────────────────────────────
conda env create -f env.yml && conda activate cs224n_dfp

# ── Step 1: Verify core implementation ───────────────────
python optimizer_test.py
python sanity_check.py

# ── Step 2: Sentiment classification ─────────────────────
python classifier.py --use_gpu --epochs 10 --lr 1e-3 --fine-tune-mode last-linear-layer
python classifier.py --use_gpu --epochs 10 --lr 1e-5 --fine-tune-mode full-model

# ── Step 3: Paraphrase detection ─────────────────────────
python paraphrase_detection.py --use_gpu --epochs 10 --lr 1e-5 --model_size gpt2

# ── Step 4: Sonnet generation (base, 100 epochs) ────────
python sonnet_generation.py --use_gpu --epochs 100 --lr 1e-5 \
    --model_size gpt2-medium --temperature 0.7 --top_p 0.9

# ── Step 5: Instruction tuning (on top of Step 4) ───────
python instruct_tune.py --use_gpu \
    --model_path 99_100-1e-05-sonnet.pt \
    --train_path data/001_first_1000.jsonl \
    --epochs 100 --lr 1e-5 --batch_size 4 \
    --model_size gpt2-medium --temperature 0.85 --top_p 0.9
```

> **Reproducibility note:** All scripts call `seed_everything(11711)` to fix random seeds across Python, NumPy, and PyTorch (including CUDA). Deterministic cuDNN is enabled via `torch.backends.cudnn.deterministic = True`.

---

## Using the Released Models

### Download from Hugging Face

```bash
# Install the HuggingFace CLI (if not already installed)
pip install huggingface-hub

# ── Download the base sonnet model (4.88 GB) ─────────────
huggingface-cli download nabin2004/WheelSonnet2-355M --local-dir ./checkpoints/WheelSonnet2-355M

# ── Download the instruction-tuned model (4.88 GB) ───────
huggingface-cli download nabin2004/WheelSonnet2-355M-it --local-dir ./checkpoints/WheelSonnet2-355M-it

# ── Download the ONNX model (652.5 MB) ───────────────────
huggingface-cli download nabin2004/WheenSonnet-355M-onnx --local-dir ./checkpoints/WheenSonnet-355M-onnx

# ── Download the quantized ONNX model (164 MB) ───────────
huggingface-cli download nabin2004/WheenSonnet-355M-quant-onnx --local-dir ./checkpoints/WheenSonnet-355M-quant-onnx
```

### Loading a Checkpoint for Inference

```python
import torch
from sonnet_generation import SonnetGPT, add_arguments

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained checkpoint
saved = torch.load(
    'checkpoints/WheelSonnet2-355M/99_100-1e-05-sonnet.pt',
    map_location=device, weights_only=False
)
model = SonnetGPT(saved['args'])
model.load_state_dict(saved['model'])
model = model.to(device)
model.eval()

# Generate a sonnet from a prompt (first 3 lines)
prompt = (
    "Shall I compare thee to a summer's day?\n"
    "Thou art more lovely and more temperate:\n"
    "Rough winds do shake the darling buds of May,\n"
)
encoding = model.tokenizer(prompt, return_tensors='pt').to(device)
_, generated_text = model.generate(encoding['input_ids'], temperature=0.7, top_p=0.9)
print(generated_text)
```

### Loading the Instruction-Tuned Model

```python
import torch
from instruct_tune import SonnetGPT, load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, optimizer, args = load_model(
    'checkpoints/WheelSonnet2-355M-it/69_100-1e-05-instruct-sonnet.pt', device
)
model.eval()

# Prompt the model with an instruction
prompt = "User: Write a sonnet about the sea.\nAssistant:"
response = model.generate(prompt, temperature=0.85, top_p=0.9)
print(response)
```

### Running ONNX Inference

```bash
pip install onnxruntime  # or onnxruntime-gpu for GPU support
```

```python
import numpy as np
import onnxruntime as ort
from transformers import GPT2Tokenizer

# Load tokenizer and ONNX session
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
session = ort.InferenceSession(
    'checkpoints/WheenSonnet-355M-quant-onnx/sonnet_gpt2_quant.onnx'
)

# Tokenize input
text = "Shall I compare thee to a summer's day?\n"
inputs = tokenizer(text, return_tensors='np')

# Run inference
logits = session.run(None, {
    'input_ids': inputs['input_ids'].astype(np.int64),
    'attention_mask': inputs['attention_mask'].astype(np.int64)
})[0]

# Get next token prediction
next_token_id = np.argmax(logits[0, -1, :])
print(f"Next token: {tokenizer.decode([next_token_id])}")
```

---

## Evaluation Metrics

| Task | Metric | Description |
|------|--------|-------------|
| Sentiment (SST-5) | **Accuracy**, **Macro-F1** | 5-class classification via `sklearn.metrics` |
| Sentiment (CFIMDB) | **Accuracy**, **Macro-F1** | Binary classification via `sklearn.metrics` |
| Paraphrase Detection | **Accuracy**, **Macro-F1** | Binary yes/no via `sklearn.metrics` |
| Sonnet Generation | **chrF** | Character-level F-score against ground truth ([Popović, 2015](https://aclanthology.org/W15-3049/)) via `sacrebleu` |

To evaluate generated sonnets:

```python
from evaluation import test_sonnet

chrf_score = test_sonnet(
    test_path='predictions/generated_sonnets.txt',
    gold_path='data/TRUE_sonnets_held_out_dev.txt'
)
print(f"chrF Score: {chrf_score:.2f}")
```

---

## Citation & Acknowledgements

This project is adapted from a prior year's CS 224N project [Implement BERT](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/project/default-final-project-handout-minbert-spr2024-updated.pdf).

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

If you use the released WheelSonnet models, please cite:

```bibtex
@misc{wheelsonnet2025,
  author       = {Nabin Oli},
  title        = {WheelSonnet: GPT-2 Fine-Tuned for Shakespearean Sonnet Generation},
  year         = {2025},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/nabin2004/WheelSonnet2-355M}},
}
```

**Key references:**
- Radford, A., et al. (2019). [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). OpenAI.
- Kingma, D. P., & Ba, J. (2014). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). ICLR 2015.
- Loshchilov, I., & Hutter, F. (2019). [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101). ICLR 2019.
- Popović, M. (2015). [chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/). WMT 2015.

---

## License

This project is licensed under the [Apache License 2.0](./LICENSE).
