# NitroGen Finetuner 🎮

[![CI](https://github.com/artryazanov/nitrogen-finetuner/actions/workflows/ci.yml/badge.svg)](https://github.com/artryazanov/nitrogen-finetuner/actions/workflows/ci.yml)
[![Lint](https://github.com/artryazanov/nitrogen-finetuner/actions/workflows/lint.yml/badge.svg)](https://github.com/artryazanov/nitrogen-finetuner/actions/workflows/lint.yml)
[![Docker Build](https://github.com/artryazanov/nitrogen-finetuner/actions/workflows/docker.yml/badge.svg)](https://github.com/artryazanov/nitrogen-finetuner/actions/workflows/docker.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project implements a **Universal Fine-Tuning Pipeline** for the **[NVIDIA NitroGen](https://huggingface.co/nvidia/NitroGen)** model. It is designed to work with any gameplay dataset generated via the NitroGen/BizHawk toolchain, supporting the model's **Diffusion Transformer (DiT)** architecture.

## 🌟 Key Features

- **Universal Architecture**: Automatically handles datasets from any console (NES, SNES, PS3, Xbox) using a dynamic gamepad serialization strategy.
- **DiT Support**: Specifically engineered for NitroGen's Diffusion Transformer architecture, which predicts a time-horizon of actions from a single frame.
- **Continuous Action Space**: Maps inputs to a rigorous 21-dimensional continuous vector space (Analogs + Buttons).
- **LoRA Integration**: Uses Low-Rank Adaptation (PEFT) for efficient fine-tuning on consumer hardware.

## 🧠 Model Architecture Details

NitroGen is a Vision-to-Action model.
- **Input**: 256x256 RGB Image (SigLip2 encoding).
- **Output**: A tensor of shape `(16, 21)` representing 16 future steps of gamepad actions.
- **Action Vector (21 dims)**:
    - 0-1: Left Stick (X, Y)
    - 2-3: Right Stick (X, Y)
    - 4-20: 17 Boolean Buttons (Binary 0.0/1.0)

## 📦 Installation

```bash
pip install .
```

## 🚀 Usage

To start fine-tuning:

```bash
python src/train.py
```

## 🧪 Running Tests

To run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src
```

### 🐳 Docker Usage

To run the training in a container:

1.  **Build**:
    ```bash
    docker build -t nitrogen-finetuner .
    ```

2.  **Run (with GPU support)**:
    ```bash
    docker run --gpus all -v $(pwd)/models:/app/models nitrogen-finetuner
    ```


### Configuration
Hyperparameters are defined in `src/config.py`. You can adjust:
- `dataset_name`: The Hugging Face dataset ID.
- `prediction_horizon`: How many future frames to predict (Default: 16).
- `analog_deadzone`: The threshold for filtering stick noise.


## 🛠️ NitroGen/BizHawk Toolchain

This project is part of a larger ecosystem for training AI agents on retro games. To generate compliant datasets, use the **[NitroGen BizHawk Dataset Generator](https://github.com/artryazanov/nitrogen-bizhawk-dataset-generator)**.

1.  **Export**: It runs a Lua script inside the **[BizHawk](https://tasvideos.org/BizHawk)** emulator to record gameplay frames and controller inputs from `.bk2` movie files.
2.  **Convert**: A Python script processes this raw data into the required Parquet format (images embedded + action vectors).

This toolchain ensures that images are pre-processed (cropped/padded to 256x256) and actions are correctly mapped for the `UniversalVectorProcessor`.

## 📊 Dataset Requirements
The pipeline expects a Hugging Face dataset (parquet) with columns matching standard gamepad inputs:
- `image`: Image column.
- `south`, `east`, `north`, `west`, etc. (Booleans).
- `j_left`, `j_right` (Vector/float sequences for sticks).

The `UniversalVectorProcessor` will automatically map these fields to the 21-dim architecture.

## 📝 License

This project is licensed under the **[MIT License](LICENSE)**.

**Note on Third-Party Assets**:
- **NVIDIA NitroGen Model**: Governed by the [NVIDIA License](https://huggingface.co/nvidia/NitroGen).
- **Datasets**: The default `Felix the Cat` dataset is released under the **MIT License**. Check individual dataset licenses when using others.