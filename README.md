# NitroGen Finetuner 🎮

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

### 🐳 Docker Usage

To run the training in a container:

1.  **Build**:
    ```bash
    docker build -t nitrogen-finetuner .
    ```

2.  **Run (with GPU support)**:
    ```bash
    docker run --gpus all -v $(pwd)/checkpoints:/app/checkpoints nitrogen-finetuner
    ```


### Configuration
Hyperparameters are defined in `src/config.py`. You can adjust:
- `dataset_name`: The Hugging Face dataset ID.
- `prediction_horizon`: How many future frames to predict (Default: 16).
- `analog_deadzone`: The threshold for filtering stick noise.

## 📊 Dataset Requirements
The pipeline expects a Hugging Face dataset (parquet) with columns matching standard gamepad inputs:
- `image`: Image column.
- `south`, `east`, `north`, `west`, etc. (Booleans).
- `j_left`, `j_right` (Vector/float sequences for sticks).

The `UniversalVectorProcessor` will automatically map these fields to the 21-dim architecture.

## 📝 License

This project is licensed under the **MIT License**.

**Note on Third-Party Assets**:
- **NVIDIA NitroGen Model**: Governed by the [NVIDIA License](https://huggingface.co/nvidia/NitroGen).
- **Datasets**: The default `Felix the Cat` dataset is released under the **MIT License**. Check individual dataset licenses when using others.