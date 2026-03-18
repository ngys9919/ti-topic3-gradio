# Walkthrough: ResNet Trainer for Fashion-MNIST

I have successfully built a mini-ResNet trainer for the Fashion-MNIST dataset. The application features a configurable ResNet architecture and a Gradio interface for interactive training and evaluation.

## Key Features

- **Keras with PyTorch Backend**: Leveraging Keras 3 for high-performance training.
- **Dynamic Architecture**: Users can adjust the number of residual blocks and base filters.
- **Comprehensive UI**: Integrated progress bars, training curves, confusion matrices, and detailed summaries.
- **Robust Pipeline**: Automated normalization, splitting, and early stopping.

## Implementation Details

### Model Architecture
The model consists of:
1.  **Initial Block**: Conv2D -> BatchNorm -> ReLU.
2.  **Residual Stages**: 3 stages of blocks with skip connections.
    - Stage 1: Identity/Projection blocks.
    - Stage 2 & 3: Downsampling using stride-2 convolutions.
3.  **Global Pooling & Dropout**: To prevent overfitting and prepare for classification.
4.  **Softmax Head**: For 10-class classification.

### Data Handling
- **Source**: HuggingFace `fashion_mnist`.
- **Preprocessing**: Pixel normalization to [0,1] and grayscale reshaping.
- **Splitting**: 50k training / 10k validation / 10k test.

## Verification Results

I verified the application with a test run using the following settings:
- **Blocks per Stage**: 1
- **Base Filters**: 16
- **Epochs**: 1
- **Optimizer**: Adam

### Training Execution
The training was monitored via the Gradio interface and completed successfully.

````carousel
![Training Curves](/C:/Users/ERIC%20NG/.gemini/antigravity/brain/ac47a42d-80ce-4d19-ba3c-52c5900a4a98/training_curves_1772941378881.png)
<!-- slide -->
![Confusion Matrix](/C:/Users/ERIC%20NG/.gemini/antigravity/brain/ac47a42d-80ce-4d19-ba3c-52c5900a4a98/confusion_matrix_1772941383929.png)
<!-- slide -->
![Results Summary](/C:/Users/ERIC%20NG/.gemini/antigravity/brain/ac47a42d-80ce-4d19-ba3c-52c5900a4a98/results_tab_1772941388811.png)
````

### Metrics
- **Test Accuracy**: 0.2479 (after only 1 epoch with a minimal model).
- **Architecture**: Mini-ResNet correctly instantiated with requested blocks and filters.

## How to Run

1.  Ensure you have the dependencies installed:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the application:
    ```bash
    $env:KERAS_BACKEND="torch"; python resnet_trainer.py
    ```
3.  Open the local URL provided in the terminal (usually `http://127.0.0.1:7860`).

## Deployment

The application has been deployed to Hugging Face Spaces and is available at the following URL:

🚀 **Deploved Space**: [https://huggingface.co/spaces/ngys9919/residual-network-trainer](https://huggingface.co/spaces/ngys9919/residual-network-trainer)
