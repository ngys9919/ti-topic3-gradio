import os

# Set Keras backend to PyTorch
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, models, ops
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import gradio as gr
from datasets import load_dataset
import torch

def load_and_preprocess_data():
    """Loads Fashion-MNIST from HuggingFace and preprocesses it."""
    dataset = load_dataset("fashion_mnist")
    
    # Extract images and labels
    x_train_orig = np.array(dataset["train"]["image"])
    y_train_orig = np.array(dataset["train"]["label"])
    x_test = np.array(dataset["test"]["image"])
    y_test = np.array(dataset["test"]["label"])
    
    # Normalize and reshape
    x_train_orig = x_train_orig.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train_orig = np.expand_dims(x_train_orig, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # Split train into train and validation (50k/10k)
    indices = np.random.permutation(len(x_train_orig))
    train_indices = indices[:50000]
    val_indices = indices[50000:]
    
    x_train = x_train_orig[train_indices]
    y_train = y_train_orig[train_indices]
    x_val = x_train_orig[val_indices]
    y_val = y_train_orig[val_indices]
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def residual_block(x, filters, kernel_size=3, stride=1):
    """A standard residual block with skip connection."""
    shortcut = x
    
    # Path 1
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, kernel_size, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    # Shortcut path
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet(input_shape, num_classes, base_filters, blocks_per_stage):
    """Builds a mini-ResNet model."""
    inputs = layers.Input(shape=input_shape)
    
    # Initial Conv
    x = layers.Conv2D(32, 3, strides=1, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Stage 1
    filters = base_filters
    for _ in range(blocks_per_stage):
        x = residual_block(x, filters, stride=1)
        
    # Stage 2
    filters *= 2
    x = residual_block(x, filters, stride=2) # Downsample
    for _ in range(blocks_per_stage - 1):
        x = residual_block(x, filters, stride=1)
        
    # Stage 3
    filters *= 2
    x = residual_block(x, filters, stride=2) # Downsample
    for _ in range(blocks_per_stage - 1):
        x = residual_block(x, filters, stride=1)
        
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return models.Model(inputs, outputs)

def train_model(optimizer_name, learning_rate, blocks_per_stage, base_filters, epochs, batch_size, use_early_stopping, progress=gr.Progress()):
    """Trains the model and returns results and plots."""
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    
    model = build_resnet((28, 28, 1), 10, base_filters, blocks_per_stage)
    
    # Select optimizer
    if optimizer_name == "Adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == "SGD + Momentum":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name == "RMSprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == "AdamW":
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    callbacks = []
    if use_early_stopping:
        callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True))
    
    # Custom callback to update Gradio progress bar
    class GradioProgressCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress((epoch + 1) / epochs, desc=f"Epoch {epoch+1}/{epochs}")
            
    callbacks.append(GradioProgressCallback())
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]
    overfit_gap = train_acc - val_acc
    
    # Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy Plot
    ax1.plot(history.history["accuracy"], label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    
    # Loss Plot
    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    
    plt.tight_layout()
    plot_path = "training_curves.png"
    plt.savefig(plot_path)
    plt.close()
    
    # Confusion Matrix
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    
    # Summary
    summary = f"""
    ### Training Summary
    - **Architecture**: Mini-ResNet ({blocks_per_stage} blocks/stage, {base_filters} base filters)
    - **Configuration**: {optimizer_name} (LR: {learning_rate}), BS: {batch_size}, Epochs: {len(history.history['loss'])}
    - **Test Accuracy**: {test_acc:.4f}
    - **Training Accuracy**: {train_acc:.4f}
    - **Validation Accuracy**: {val_acc:.4f}
    - **Overfit Gap**: {overfit_gap:.4f}
    """
    
    return plot_path, cm_path, summary

# Gradio Interface
with gr.Blocks(title="Fashion-MNIST ResNet Explorer") as demo:
    gr.Markdown("# 👗 Fashion-MNIST ResNet Model Explorer")
    gr.Markdown("Train a custom Mini-ResNet and see how different parameters affect performance.")
    
    with gr.Row():
        with gr.Column(scale=1):
            optimizer = gr.Dropdown(choices=["Adam", "SGD", "SGD + Momentum", "RMSprop", "AdamW"], value="Adam", label="Optimizer")
            lr = gr.Slider(minimum=0.0001, maximum=0.1, value=0.001, step=0.0001, label="Learning Rate")
            blocks = gr.Slider(minimum=1, maximum=4, value=2, step=1, label="Residual Blocks per Stage")
            filters = gr.Dropdown(choices=[16, 32, 48, 64], value=32, label="Base Filters")
            epochs = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Epochs")
            batch_size = gr.Slider(minimum=16, maximum=512, value=128, step=16, label="Batch Size")
            early_stop = gr.Checkbox(value=True, label="Enable Early Stopping")
            train_btn = gr.Button("🚀 Start Training", variant="primary")
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Training Curves"):
                    output_plot = gr.Image(label="Accuracy & Loss")
                with gr.TabItem("Confusion Matrix"):
                    output_cm = gr.Image(label="Matrix")
                with gr.TabItem("Results"):
                    output_summary = gr.Markdown()
    
    train_btn.click(
        fn=train_model,
        inputs=[optimizer, lr, blocks, filters, epochs, batch_size, early_stop],
        outputs=[output_plot, output_cm, output_summary]
    )

if __name__ == "__main__":
    demo.launch()
