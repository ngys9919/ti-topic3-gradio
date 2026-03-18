# Implementation Plan: ResNet Trainer for Fashion-MNIST

Create a comprehensive Python application for training and exploring ResNet models on the Fashion-MNIST dataset using Keras (PyTorch backend) and Gradio.

## Proposed Changes

### [Core Application]

#### [NEW] [resnet_trainer.py](file:///d:/source/ti-vibe-coding-ai/gradio-topic3/resnet_trainer.py)

- **Backend Configuration**: Set `KERAS_BACKEND="torch"` before importing Keras.
- **Data Pipeline**:
    - Load Fashion-MNIST using HuggingFace's `datasets` library.
    - Normalize pixel values to [0, 1].
    - Reshape images to [(28, 28, 1)](file:///d:/source/ti-vibe-coding-ai/gradio-topic3/resnet_trainer.py#97-196).
    - Split the training data (60k) into 50,000 for training and 10,000 for validation.
- **Model Architecture**:
    - Implement a parameterized ResNet builder.
    - **Initial Layer**: Conv2D (32 filters, 3x3), BatchNorm, ReLU.
    - **Stages**: 3 stages of residual blocks.
        - Stage 1: Base filters (input parameter), stride 1.
        - Stage 2: 2x Base filters, stride 2 (downsampling).
        - Stage 3: 4x Base filters, stride 2 (downsampling).
    - **Residual Block**:
        - Path 1: Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm.
        - Path 2 (Shortcut): identity if filters/strides match, else 1x1 Conv2D + BatchNorm.
        - Merge: Add(Path 1, Path 2) -> ReLU.
    - **Head**: GlobalAveragePooling2D, Dropout(0.3), Dense(10, softmax).
- **Gradio UI**:
    - **Inputs**:
        - Optimizer (Adam, SGD, SGD + Momentum, RMSprop, AdamW).
        - Learning Rate (0.0001 to 0.1).
        - Blocks per Stage (1 to 4).
        - Base Filters (16, 32, 48, 64).
        - Epochs (default 10).
        - Batch Size (default 128).
        - Early Stopping (Checkbox).
    - **Outputs**:
        - Training progress bar (using Gradio's `gr.Progress`).
        - Accuracy/Loss plots.
        - Confusion Matrix (calculated on the test set).
        - Training summary text.
- **Removal of Flagging**: Disable the Gradio flag button.

### [Deployment]

#### [NEW] [deploy_to_hf.py](file:///d:/source/ti-vibe-coding-ai/gradio-topic3/deploy_to_hf.py)
- **Script Purpose**: Automate the creation and upload of the Gradio app to Hugging Face Spaces.
- **Dependencies**: Use `huggingface_hub` and [io](file:///d:/source/ti-vibe-coding-ai/gradio-topic3/resnet_trainer.py#122-125).
- **Steps**:
    - Authenticate using the provided token.
    - Create a new Space named `residual-network-trainer`.
    - Upload [resnet_trainer.py](file:///d:/source/ti-vibe-coding-ai/gradio-topic3/resnet_trainer.py) as `app.py`.
    - Upload [requirements.txt](file:///d:/source/ti-vibe-coding-ai/gradio-topic3/requirements.txt).
    - Print the Space URL.

## Verification Plan

### Automated Tests
- Run the script and verify that it starts without errors.
- Train a small model (1 block per stage, 16 filters, 1 epoch) to verify the full pipeline (data loading -> model building -> training -> evaluation -> UI output).

### Manual Verification
- **Model Parameters**: Verify the number of parameters changes correctly when adjusting "Blocks per Stage" and "Base Filters".
- **Optimizer Check**: Verify that different optimizers are correctly instantiated and used during training.
- **UI Responsiveness**: Ensure the progress bar updates in real-time.
- **Plot Correctness**: Verify that loss/accuracy plots reflect the training progress and that the confusion matrix correctly labels the 10 Fashion-MNIST classes.
- **Early Stopping**: Verify that training stops early if the checkbox is checked and validation loss stops improving.
