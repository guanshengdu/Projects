# %%

import numpy as np

X_t = np.load(
    "/home/gs/code/AS24-V59/project/data/BCICIV_2a_gdf/X_bci_2a_A01T.npy"
)


y_t = np.load(
    "/home/gs/code/AS24-V59/project/data/BCICIV_2a_gdf/y_bci_2a_A01T.npy"
)

y_t = y_t - 1


X_e = np.load("/home/gs/code/AS24-V59/project/data/BCICIV_2a_gdf/X_bci_2a_A01E.npy")

y_e = np.load("/home/gs/code/AS24-V59/project/data/BCICIV_2a_gdf/y_bci_2a_A01E.npy")

y_e = y_e - 1


# %%

import numpy as np

# z:\home\gs\code\AS24-V59\project\data\BCICIV_2a_gdf\X_bci_2a_training_data.npy
# z:\home\gs\code\AS24-V59\project\data\BCICIV_2a_gdf\X_bci_2a_evaluation_data_2.npy

X_t = np.load(
    "/home/gs/code/AS24-V59/project/data/BCICIV_2a_gdf/X_bci_2a_training_data.npy"
)


y_t = np.load(
    "/home/gs/code/AS24-V59/project/data/BCICIV_2a_gdf/y_bci_2a_training_data.npy"
)

X_e = np.load(
    "/home/gs/code/AS24-V59/project/data/BCICIV_2a_gdf/X_bci_2a_evaluation_data_2.npy"
)

y_e = np.load(
    "/home/gs/code/AS24-V59/project/data/BCICIV_2a_gdf/y_bci_2a_evaluation_data_2.npy"
)


# Event ID 768: Start of a trial
# Event ID 769: Cue onset left (class 1)
# Event ID 770: Cue onset right (class 2)
# Event ID 771: Cue onset foot (class 3)
# Event ID 772: Cue onset tongue (class 4)
# Event ID 783: Cue onset unknown

# Mapping of event IDs to new labels
event_mapping = {"769": 0, "770": 1, "771": 2, "772": 3}

# Replace event IDs in y_t with new labels
y_t = np.array([event_mapping.get(event, event) for event in y_t])

y_e = np.array([event_mapping.get(event, event) for event in y_e])

# X_t.shape, y_t.shape


# %%
# the one inspired from EEG Conformer

import torch
import torch.nn as nn

# %%
class ConvModule(nn.Module):
    def __init__(
        self,
        num_channels,
        k=40,
        kernel_size=(1, 25),
        pooling_size=(1, 75),
        pooling_stride=(1, 15),
    ):
        super(ConvModule, self).__init__()
        self.temporal_conv = nn.Conv2d(1, k, kernel_size=kernel_size, stride=(1, 1))
        self.spatial_conv = nn.Conv2d(
            k, k, kernel_size=(num_channels, 1), stride=(1, 1)
        )
        self.batch_norm = nn.BatchNorm2d(k)
        self.activation = nn.ELU()
        self.pooling = nn.AvgPool2d(kernel_size=pooling_size, stride=pooling_stride)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.activation(self.batch_norm(x))
        x = self.spatial_conv(x)
        x = self.activation(self.batch_norm(x))
        x = self.pooling(x)
        x = self.dropout(x)
        x = x.squeeze(2).permute(0, 2, 1)
        return x


# %%

# Version 2
class ConvModule(nn.Module):
    def __init__(
        self,
        num_channels,
        k=40,  # Number of kernels
        kernel_size=(1, 25),  # Temporal kernel size
        pooling_size=(1, 75),  # Pooling kernel size
        pooling_stride=(1, 15),  # Pooling stride
    ):
        super(ConvModule, self).__init__()
        # Temporal convolution
        self.temporal_conv = nn.Conv2d(1, k, kernel_size=kernel_size, stride=(1, 1))
        # Spatial convolution across channels
        self.spatial_conv = nn.Conv2d(
            k, k, kernel_size=(num_channels, 1), stride=(1, 1)
        )
        # Batch normalization and activation
        self.batch_norm = nn.BatchNorm2d(k)
        self.activation = nn.ELU()
        # Pooling and dropout
        self.pooling = nn.AvgPool2d(kernel_size=pooling_size, stride=pooling_stride)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Forward pass for ConvModule.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 1, num_channels, timepoints).

        Returns:
            Tensor: Processed tensor of shape (batch_size, seq_len, k).
        """
        # Transpose to match paper's data shape: (batch_size, 22, 1, 1000)
        x = x.permute(0, 2, 1, 3)  # New shape: (batch_size, 22, 1, 1000)

        # Apply temporal convolution
        x = self.temporal_conv(x)
        x = self.activation(self.batch_norm(x))

        # Apply spatial convolution
        x = self.spatial_conv(x)
        x = self.activation(self.batch_norm(x))

        # Pooling and dropout
        x = self.pooling(x)
        x = self.dropout(x)

        # Reshape output for attention module
        x = x.squeeze(2).permute(0, 2, 1)  # Shape: (batch_size, seq_len, k)
        return x


# %%

# Version 2

import numpy as np


def segment_and_reconstruct(X, y, num_segments=8):
    """
    Segments trials into smaller parts, shuffles them, and reconstructs them for augmentation.

    Args:
        X (numpy array): EEG data of shape (num_samples, 1, num_channels, timepoints).
        y (numpy array): Labels of shape (num_samples,).
        num_segments (int): Number of segments per trial.

    Returns:
        X_aug (numpy array): Augmented EEG data.
        y_aug (numpy array): Corresponding labels.
    """
    X_aug, y_aug = [], []
    segment_size = X.shape[-1] // num_segments

    for i, label in enumerate(y):
        trial = X[i]
        segments = [
            trial[:, :, j * segment_size : (j + 1) * segment_size]
            for j in range(num_segments)
        ]

        # Shuffle segments and reconstruct the trial
        np.random.shuffle(segments)
        reconstructed = np.concatenate(segments, axis=-1)

        X_aug.append(reconstructed)
        y_aug.append(label)

    return np.array(X_aug), np.array(y_aug)


# Example Usage
X_aug, y_aug = segment_and_reconstruct(X_t, y_t, num_segments=8)

# Combine augmented data with the original dataset
X_combined = np.concatenate((X_t, X_aug))
y_combined = np.concatenate((y_t, y_aug))

# %%

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        """
        Classifier Module for EEG classification.

        Args:
            input_dim (int): The dimensionality of the input features.
            num_classes (int): Number of output classes.
        """
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),  # First fully connected layer
            nn.ELU(),  # Non-linear activation
            nn.Dropout(0.4),  # Regularization
            nn.Linear(256, 128),  # Second fully connected layer
            nn.ELU(),  # Non-linear activation
            nn.Dropout(0.4),  # Regularization
            nn.Linear(128, num_classes),  # Output layer
        )

    def forward(self, x):
        """
        Forward pass through the classifier.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes).
        """
        return self.fc(x)


# the one inspired from EEG Conformer


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, V)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        attention = self.attention(Q, K, V)
        attention = (
            attention.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_k)
        )
        return self.dropout(self.out(attention))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention with residual connection
        attn_output = self.mha(x)
        x = self.norm1(x + attn_output)
        # Feed-forward network with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x


class SelfAttentionModule(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(SelfAttentionModule, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# The one inspired from EEG Conformer


# The one inspired from EEG Conformer

# %%
class EEGModel(nn.Module):
    def __init__(self, num_channels, d_model, num_classes, num_heads, num_layers, d_ff):
        super(EEGModel, self).__init__()
        self.conv_module = ConvModule(num_channels=num_channels, k=d_model)
        self.self_attention = SelfAttentionModule(
            d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers
        )
        self.classifier = Classifier(input_dim=d_model, num_classes=num_classes)

    def forward(self, x):
        x = self.conv_module(x)  # Shape: (batch_size, seq_len, d_model)
        x = self.self_attention(x)  # Shape: (batch_size, seq_len, d_model)
        x = x.mean(dim=1)  # Global Average Pooling (GAP)
        x = self.classifier(x)  # Shape: (batch_size, num_classes)
        return x


# %%

# Version 2
class EEGModel(nn.Module):
    def __init__(self, num_channels, d_model, num_classes, num_heads, num_layers, d_ff):
        super(EEGModel, self).__init__()
        self.conv_module = ConvModule(num_channels=num_channels, k=d_model)
        self.self_attention = SelfAttentionModule(
            d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers
        )
        self.classifier = Classifier(input_dim=d_model, num_classes=num_classes)

    def forward(self, x):
        x = self.conv_module(x)  # Shape: (batch_size, seq_len, d_model)
        x = self.self_attention(x)  # Shape: (batch_size, seq_len, d_model)
        x = x.mean(dim=1)  # Global Average Pooling
        x = self.classifier(x)  # Shape: (batch_size, num_classes)
        return x


# %%
import torch
from torch.utils.data import Dataset, DataLoader


class EEGDataset(Dataset):
    def __init__(self, X, y):
        """
        Custom Dataset for EEG data.
        Args:
            X (Tensor): Input data of shape (num_samples, channels, timepoints).
            y (Tensor): Labels of shape (num_samples,).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

from sklearn.metrics import accuracy_score, cohen_kappa_score


def evaluate_model(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)

    return {"loss": total_loss / len(dataloader), "accuracy": accuracy, "kappa": kappa}

# %%
# Create dataset and data loaders
batch_size = 32

dataset = EEGDataset(X_t, y_t)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Create DataLoader for evaluation
eval_dataset = EEGDataset(X_e, y_e)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)


# Example parameters
num_channels = X_t.shape[2]  # Number of EEG channels
d_model = 40  # Embedding size (output of conv module)
num_classes = len(set(y_t))  # Number of EEG categories
num_heads = 10  # Number of attention heads
num_layers = 6  # Number of self-attention layers
d_ff = 128  # Hidden size for feed-forward layers

# Instantiate the model
model = EEGModel(
    num_channels=num_channels,
    d_model=d_model,
    num_classes=num_classes,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
)

# The one inspired from EEG Conformer

import torch.optim as optim

# Loss function: Cross-Entropy Loss
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam with author-recommended settings
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0002,  # Learning rate from the paper
    betas=(0.5, 0.999),  # Specific β values for Adam
    weight_decay=1e-5,  # L2 regularization
)

# Learning Rate Scheduler (optional but helps stabilization)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=500, gamma=0.1
)  # Reduce LR every 500 steps


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to GPU if available



# Training loop
epochs = 2000

loss_history = []


train_loss_history = []
train_accuracy_history = []
train_f1_history = []
train_precision_history = []

eval_loss_history = []
eval_accuracy_history = []
eval_hamming_loss_history = []


# %%
from sklearn.metrics import accuracy_score, f1_score, precision_score, hamming_loss

# Training Loop
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    all_preds, all_labels = [], []

    for inputs, labels in train_loader:
        # Move data to device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

        # Track predictions and labels
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Compute training metrics
    train_loss = running_loss / len(train_loader)
    train_accuracy = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average="weighted")
    train_precision = precision_score(all_labels, all_preds, average="weighted")

    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)
    train_f1_history.append(train_f1)
    train_precision_history.append(train_precision)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f} | "
        f"Train F1: {train_f1:.4f} | "
        f"Train Precision: {train_precision:.4f}"
    )

    # Evaluate every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()  # Set model to evaluation mode
        eval_loss = 0.0
        eval_preds, eval_labels = [], []

        with torch.no_grad():
            for eval_inputs, eval_labels_batch in eval_loader:
                eval_inputs, eval_labels_batch = eval_inputs.to(
                    device
                ), eval_labels_batch.to(device)

                # Forward pass
                eval_outputs = model(eval_inputs)
                eval_loss += criterion(eval_outputs, eval_labels_batch).item()

                # Track predictions and labels
                eval_preds_batch = torch.argmax(eval_outputs, dim=1)
                eval_preds.extend(eval_preds_batch.cpu().numpy())
                eval_labels.extend(eval_labels_batch.cpu().numpy())

        # Compute evaluation metrics
        eval_loss /= len(eval_loader)
        eval_accuracy = accuracy_score(eval_labels, eval_preds)
        eval_hamming_loss = hamming_loss(eval_labels, eval_preds)

        eval_loss_history.append(eval_loss)
        eval_accuracy_history.append(eval_accuracy)
        eval_hamming_loss_history.append(eval_hamming_loss)

        print(
            f"--- Evaluation --- Epoch {epoch+1}/{epochs} | "
            f"Eval Loss: {eval_loss:.4f} | "
            f"Eval Accuracy: {eval_accuracy:.4f} | "
            f"Eval Hamming Loss: {eval_hamming_loss:.4f}"
        )

# %%

# Version 2

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    # Augment data every few epochs
    if epoch % 50 == 0:
        X_aug, y_aug = segment_and_reconstruct(X_t, y_t, num_segments=8)
        X_combined = np.concatenate((X_t, X_aug))
        y_combined = np.concatenate((y_t, y_aug))

        # Update DataLoader with augmented data
        dataset = EEGDataset(X_combined, y_combined)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero gradients

        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss

        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Compute training metrics
    train_loss = running_loss / len(train_loader)
    train_accuracy = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average="weighted")
    train_precision = precision_score(all_labels, all_preds, average="weighted")

    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)
    train_f1_history.append(train_f1)
    train_precision_history.append(train_precision)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f} | "
        f"Train F1: {train_f1:.4f} | "
        f"Train Precision: {train_precision:.4f}"
    )
    train_loss = running_loss / len(train_loader)
    train_accuracy = accuracy_score(all_labels, all_preds)
    print(
        f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.4f}"
    )

    # Evaluate every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()  # Set model to evaluation mode
        eval_loss = 0.0
        eval_preds, eval_labels = [], []

        with torch.no_grad():
            for eval_inputs, eval_labels_batch in eval_loader:
                eval_inputs, eval_labels_batch = eval_inputs.to(
                    device
                ), eval_labels_batch.to(device)

                # Forward pass
                eval_outputs = model(eval_inputs)
                eval_loss += criterion(eval_outputs, eval_labels_batch).item()

                # Track predictions and labels
                eval_preds_batch = torch.argmax(eval_outputs, dim=1)
                eval_preds.extend(eval_preds_batch.cpu().numpy())
                eval_labels.extend(eval_labels_batch.cpu().numpy())

        # Compute evaluation metrics
        eval_loss /= len(eval_loader)
        eval_accuracy = accuracy_score(eval_labels, eval_preds)
        eval_hamming_loss = hamming_loss(eval_labels, eval_preds)

        eval_loss_history.append(eval_loss)
        eval_accuracy_history.append(eval_accuracy)
        eval_hamming_loss_history.append(eval_hamming_loss)

        print(
            f"--- Evaluation --- Epoch {epoch+1}/{epochs} | "
            f"Eval Loss: {eval_loss:.4f} | "
            f"Eval Accuracy: {eval_accuracy:.4f} | "
            f"Eval Hamming Loss: {eval_hamming_loss:.4f}"
        )

# %%
import matplotlib.pyplot as plt

# Plot training and evaluation metrics
plt.figure(figsize=(12, 8))

# Plot training loss
plt.subplot(2, 2, 1)
plt.plot(train_loss_history, label="Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

# Plot training accuracy
plt.subplot(2, 2, 2)
plt.plot(train_accuracy_history, label="Train Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()

# Plot training F1 score
plt.subplot(2, 2, 3)
plt.plot(train_f1_history, label="Train F1 Score")
plt.xlabel("Epochs")
plt.ylabel("F1 Score")
plt.title("Training F1 Score")
plt.legend()

# Plot evaluation accuracy
plt.subplot(2, 2, 4)
plt.plot(eval_accuracy_history, label="Eval Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Evaluation Accuracy")
plt.legend()

plt.tight_layout()
plt.show()


# %%

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(eval_labels, eval_preds)

# Display the confusion matrix
class_names = [
    "Left",
    "Right",
    "Foot",
    "Tongue",
]  # Update these names based on your dataset

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="viridis", xticks_rotation="vertical")
plt.title("Confusion Matrix")
plt.show()


# %%
save_path = "eeg_model_and_optimizer_full_data_20241215.pth"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,  # Save the current epoch for resuming
        "loss_history": loss_history,  # Optionally save training history
    },
    save_path,
)
print(f"Model and optimizer saved to {save_path}")

# %%
load_path = "eeg_model_and_optimizer.pth"
checkpoint = torch.load(load_path)

# Load model and optimizer states
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# Load additional data
start_epoch = checkpoint["epoch"]
loss_history = checkpoint["loss_history"]

model.to(device)
model.eval()
print(f"Model and optimizer loaded from {load_path}, resuming from epoch {start_epoch}")
