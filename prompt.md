Create a single Python file for training a mini-ResNet on Fashion-MNIST using
Keras with PyTorch backend and Gradio. Load the dataset from HuggingFace with
load_dataset("fashion_mnist").

Dataset:
- Fashion-MNIST: 28x28 grayscale images, 10 classes
- Normalize to [0,1] and reshape to (28, 28, 1)
- Split training set into 50,000 train / 10,000 validation

Model Architecture:
- Input layer for 28x28x1 images
- Initial Conv2D layer with 32 filters (3x3), BatchNorm, ReLU
- 3 stages of residual blocks, doubling filters each stage (32 → 64 → 128)
- Each residual block: Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → Add(shortcut) → ReLU
- Use 1x1 convolution to project the shortcut when dimensions change
- Use stride=2 at the start of stages 2 and 3 to downsample
- GlobalAveragePooling2D → Dropout(0.3) → Dense(10, softmax)

Let users adjust via Gradio:
  - Optimizer (Adam, SGD, SGD + Momentum, RMSprop, AdamW)
  - Learning rate slider
  - Number of residual blocks per stage (1-4)
  - Base filters (16, 32, 48, 64)
  - Epochs slider (default 10)
  - Batch size slider (default 128)
  - Early stopping checkbox

- Show a progress bar during training
- Display accuracy/loss plots, confusion matrix, and a training summary
- The summary should show the architecture, configuration, test accuracy, and overfit gap
- Remove the Gradio flag button
